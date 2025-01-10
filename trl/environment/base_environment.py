# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

from typing import Optional

import torch
from accelerate.utils import extract_model_from_parallel
from transformers import StoppingCriteria, StoppingCriteriaList, DynamicCache

from ..import_utils import is_rich_available


if is_rich_available():
    from rich import print
    from rich.text import Text


class StringStoppingCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generations in the batch are completed."""

    def __init__(self, stop_strings, tokenizer):
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        self.first_call = True

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the stop strings or terminated early."""
        if self.first_call:
            self.generated_tokens = [1 for _ in range(input_ids.shape[0])]
            self.start_length = input_ids.shape[-1] - 1
            self.first_call = False
        decoded_generations = self.tokenizer.batch_decode(input_ids[:, self.start_length :])
        done = []

        for i, decoded_generation in enumerate(decoded_generations):
            sequence_complete = any(stop_string in decoded_generation for stop_string in self.stop_strings) or self.tokenizer.eos_token_id in input_ids[i, self.start_length :]
            done.append(sequence_complete)
            if not sequence_complete:
                self.generated_tokens[i] += 1

        if all(done):
            self.first_call = True

        return all(done)


class TextHistory:
    """The TextHistory class keeps track of the history of an interaction between the language model and the environment."""

    def __init__(self, text, tokens, system=True):
        """
        Initialize TextHistory.

        Args:
            text (`str`): The text of the first segment.
            tokens (`torch.LongTensor`): The tokens of the first segment.
            system (`bool`, *optional*): Whether the first segment is a system or user segment.
        """
        self.system_spans = []
        self.text_spans = []
        self.token_spans = []
        self.token_masks = torch.tensor([], dtype=torch.long).to(tokens.device)
        self.text = ""
        self.tokens = torch.tensor([], dtype=torch.long).to(tokens.device)
        self.completed = False
        self.truncated = False
        self.reward = 0.0

        self.prompt_color = "black on grey85"
        self.system_color = "black on cyan3"
        self.model_color = "black on deep_sky_blue1"
        self.reward_color = "black on plum1"

        self.append_segment(text, tokens, system=system)

    def append_segment(self, text, tokens, system=True):
        """
        Append a new segment to the history.

        Args:
            text (`str`): The text of the new segment.
            tokens (`torch.LongTensor`): The tokens of the new segment.
            system (`bool`, *optional*): Whether the new segment is a system or user segment.
        """

        if len(text) == 0 or len(tokens) == 0:
            raise ValueError("Can't append empty text or token list to history.")

        original_text_length = len(self.text)

        self.text += text
        self.text_spans.append((original_text_length, len(self.text)))
        self.system_spans.append(system)

        original_token_length = len(self.tokens)

        self.tokens = torch.cat((self.tokens, tokens))
        if system:
            self.token_masks = torch.cat((self.token_masks, torch.zeros_like(tokens)))
        else:
            self.token_masks = torch.cat((self.token_masks, torch.ones_like(tokens)))
        self.token_spans.append((original_token_length, len(self.tokens)))

    def complete(self, truncated=False):
        """
        Mark the history as completed.
        """
        self.completed = True
        self.truncated = truncated

    @property
    def last_text_segment(self):
        """
        Get the last text segment.
        """
        start, end = self.text_spans[-1]
        return self.text[start:end]
    
    @property
    def last_token_segment(self):
        """
        Get the last token segment
        """
        start, end = self.token_spans[-1]
        return self.tokens[start:end]

    def split_query_response_tokens(self):
        """
        Split the tokens into query and response tokens.
        """
        split_index = self.token_spans[0][1]
        query = self.tokens[:split_index]
        response = self.tokens[split_index:]
        mask = self.token_masks[split_index:]

        return query, response, mask

    def show_text(self, show_legend=False):
        """
        Print the text history.
        """
        if not is_rich_available():
            raise ImportError(
                "The `rich` library is required to display text with formatting. "
                "Install it using `pip install rich`."
            )

        text = Text(self.text)
        text.stylize(self.prompt_color, self.text_spans[0][0], self.text_spans[1][0])
        for i, (start, end) in enumerate(self.text_spans[1:]):
            if self.system_spans[i + 1]:
                text.stylize(self.system_color, start, end)
            else:
                text.stylize(self.model_color, start, end)

        text.append(f"\n\nReward: {self.reward}", style=self.reward_color)
        print(text)

        if show_legend:
            self.show_colour_legend()

    def show_tokens(self, tokenizer, show_legend=False):
        """
        Print the history tokens.
        """
        if not is_rich_available():
            raise ImportError(
                "The `rich` library is required to display tokens with formatting. "
                "Install it using `pip install rich`."
            )

        text = Text()
        prompt_end = self.token_spans[0][1]
        for i, (token, mask) in enumerate(zip(self.tokens, self.token_masks)):
            if i < prompt_end:
                text.append(tokenizer.convert_ids_to_tokens(token.item()), style=self.prompt_color)
                text.append(" ")
            elif mask == 0:
                text.append(tokenizer.convert_ids_to_tokens(token.item()), style=self.system_color)
                text.append(" ")
            else:
                text.append(tokenizer.convert_ids_to_tokens(token.item()), style=self.model_color)
                text.append(" ")
        text.append(f"\n\nReward: {self.reward}", style=self.reward_color)
        print(text)
        if show_legend:
            self.show_colour_legend()

    def show_colour_legend(self):
        """
        Print the colour legend.
        """
        if not is_rich_available():
            raise ImportError(
                "The `rich` library is required to display colour legends with formatting. "
                "Install it using `pip install rich`."
            )
        text = Text("\n\n(Colour Legend: ")
        text.append("Prompt", style=self.prompt_color)
        text.append("|")
        text.append("System", style=self.system_color)
        text.append("|")
        text.append("Model", style=self.model_color)
        text.append("|")
        text.append("Reward", style=self.reward_color)
        text.append(")")
        print(text)


class TextEnvironment:
    """
    The TextEnvironment enables interaction of a LLM with an environment using tools.
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        tools=None,
        reward_fn=None,
        prompt=None,
        max_turns=4,
        max_tool_reponse=100,
        max_length=None,
        generation_kwargs=None,
        use_cache=False
    ):
        """
        Initialize TextEnvironment.

        Args:
            model (`PreTrainedModelWrapper`): The model to use for generation.
            tokenizer (`transformers.PreTrainedTokenizer`): The tokenizer to use for generation.
            tools (list): A list of tools to use for interaction.
            reward_fn (function): A function that takes a string and returns a reward.
            prompt (str): The base prompt to use for generation. Is prepended to the tasks.
            max_turns (Optional[int]): The maximum number of turns to allow.
            max_tool_response (Optional[int]): The maximum number of characters to allow in a tool response.
            max_length (Optional[int]): The maximum number of tokens to allow in an episode.
            generation_kwargs (Optional[dict]): A dictionary of keyword arguments to pass to the model's generate method.
            use_cache (bool): Whether or not to cache past_key_values between segments. When using caching, TextEnvironment is not suited for training use, i.e. backpropagation through the generated graph. Use with Trainers is of course possible. Furthermore, caching requires, that there be no calculation dependencies between examples at inference time. When using BatchNorm, the model should thus be in eval mode.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        if isinstance(tools, dict):
            self.tools = tools
        else:
            self.tools = {tool.__class__.__name__: tool for tool in tools}
        self.reward_fn = reward_fn
        self.max_length = max_length
        self.request_token = "<request>"
        self.call_token = "<call>"
        self.response_token = "<response>"
        self.submit_token = "<submit>"
        self.max_turns = max_turns
        self.max_tool_response = max_tool_reponse
        self.use_cache = use_cache

        if generation_kwargs is None:
            self.generation_kwargs = dict()
        else:
            self.generation_kwargs = generation_kwargs

        self.is_encoder_decoder = hasattr(self.model, "is_encoder_decoder")
        self.current_device = extract_model_from_parallel(self.model).pretrained_model.device

    def run(self, queries, **rewards_kwargs):
        """
        Run the environment on a list of queries.

        Args:
            queries (list[str]): A list of queries to run the model in the environment on.
        """
        turns = 0

        queries = [self.prompt + task for task in queries]
        queries_tokens = [
            self.tokenizer(query, return_tensors="pt").input_ids[0].to(self.model.pretrained_model.device)
            for query in queries
        ]

        histories = [TextHistory(q, qt, system=True) for q, qt in zip(queries, queries_tokens)]
        
        past_key_values,past_attention_masks,past_input_ids,last_active_histories = (None,None,None,None)

        while any(not history.completed for history in histories) and turns < self.max_turns:
            if self.use_cache:
                histories,past_key_values,past_attention_masks,past_input_ids,last_active_histories = self.generate(histories,past_key_values,past_attention_masks,past_input_ids,last_active_histories)
            else:
                #Discard cache
                histories,_,_,_,_ = self.generate(histories,past_key_values,past_attention_masks,past_input_ids,last_active_histories)
            histories = self.tasks_end_check(histories)
            # TODO: make this parallel rather than for-loop
            for i in range(len(histories)):
                histories[i] = self.step(histories[i])
            histories = self.tasks_end_check(histories, model_turn=False)
            turns += 1
        self.compute_reward(histories, **rewards_kwargs)

        # convert a list of (q, r, m) tuples to lists of all qs, rs, and ms respectively
        queries, responses, masks = map(list, zip(*[history.split_query_response_tokens() for history in histories]))

        rewards = [history.reward for history in histories]
        return queries, responses, masks, rewards, histories

    def step(self, history):
        """
        Step the environment forward one turn.

        Args:
            history (`TextHistory`): The history to step forward.
        """
        truncated, ended = self.task_end_check(history)
        if ended:
            history.complete(truncated=truncated)
        if history.completed:
            return history

        tool, query = self.parse_tool_call(history.last_text_segment)
        if tool is None or query is None:
            response = f"Unknown tool call: {history.last_text_segment}"
        else:
            if tool not in self.tools:
                response = f"Unknown tool {tool}."
            try:
                response = self.tools[tool](query)
            except Exception as error:
                response = f"Tool error: {str(error)}"

        if len(response) > self.max_tool_response:
            response = response[: (self.max_tool_response - 3)] + "..."

        history.append_segment(
            response + self.response_token,
            self.tokenizer(response + self.response_token, return_tensors="pt")
            .input_ids[0]
            .to(self.model.pretrained_model.device),
            system=True,
        )

        return history

    def parse_tool_call(self, text):
        """
        Parse request string. Expected format: <request><tool_name>query<call>
        """
        result = re.search(f"(?<={self.request_token}).*?(?={self.call_token})", text, re.DOTALL)

        # if we can't find a <request>/<call> span we return none
        if result is None:
            return None, None
        else:
            extracted_text = result.group()

        result = re.search(r"<(.*?)>", extracted_text)

        # if we can't find a tool name we return none
        if result is None:
            return None, None
        else:
            tool = result.group(1)

        # split off the tool name
        query = ">".join(extracted_text.split(">")[1:])

        return tool, query

    def compute_reward(self, histories, **reward_kwargs):
        """
        Compute the reward for a list of histories.
        """
        rewards = self.reward_fn([history.last_text_segment for history in histories], **reward_kwargs)
        for history, reward in zip(histories, rewards):
            history.reward = reward
        return histories

    def _next_input(self,history):
        return history.last_token_segment if not history.completed else torch.tensor([])

    def _combine_cache(self,example_mask,past_key_values,past_attention_masks,past_input_ids):
        """
        combines all caches in order to exclude completed histories from further generation

        Args:
            batch_examples (list[bool]): mask indicating for each example, whether it is supposed to remain or not
            past_key_values (list[transformers.DynamicCache]) : Batched list of caches from the last generation
            past_attention_masks (list[torch.Tensor]): Batched list of attention masks from the last generation
            past_input_ids (list[torch.Tensor]): Batched list of input ids from the last generation
        """
        legacy_format = [cache.to_legacy_cache() for cache in past_key_values ]
        example_mask_offset = 0
        combined_cache = []
        for layer_id in range(len(legacy_format[0])):
            combined_layer = None
            for cache_idx, cache in enumerate(legacy_format):
                layer = cache[layer_id]
                num_examples = len(layer[0])
                new_keys = layer[0][example_mask[example_mask_offset:example_mask_offset+num_examples]]
                new_values = layer[1][example_mask[example_mask_offset:example_mask_offset+num_examples]]
                if combined_layer is None:
                    combined_layer = (new_keys,new_values)
                else:
                    other_new_keys,other_new_values = combined_layer
                    combined_layer = (torch.concat([other_new_keys,new_keys],dim=0),torch.concat([other_new_values,new_values],dim=0))
                example_mask_offset += num_examples
            combined_cache.append(combined_layer)
        combined_cache = tuple(combined_cache)

        combined_attention_masks = torch.concat(past_attention_masks,dim=0)[example_mask]
        combined_input_ids = torch.concat(past_input_ids,dim=0)[example_mask]

        return combined_cache, combined_attention_masks, combined_input_ids

    def generate(self, histories,past_key_values=None,past_attention_masks=None,past_input_ids=None,last_active_histories=None):
        """
        Generate responses for a list of histories.
        Either all of past_key_values, past_attention_masks, past_input_ids,last_active_histories are provided or all are None.
        Args:
            histories (list[TextHistory]):
            past_key_values (Optional[list[transformers.DynamicCache]]): Batched list of caches from the last generation
            past_attention_masks (Optional[list[torch.Tensor]]): Batched list of attention masks from the last generation
            past_input_ids (Optional[list[torch.Tensor]]): Batched list of input ids from the last generation
            last_active_histories (Optional[list[int]]): indices of histories for which generation took place during the last generation turn
        """
        active_histories = [i for i in range(len(histories)) if not histories[i].completed]
        combined_past_key_values,combined_past_attention_masks, combined_past_input_ids = (None,None,None)
        
        if past_key_values is not None:
            query_tensors = [self._next_input(histories[i]) for i in active_histories]
            example_mask = [(not histories[i].completed) for i in last_active_histories]
            combined_past_key_values,combined_past_attention_masks, combined_past_input_ids = self._combine_cache(example_mask,past_key_values,past_attention_masks,past_input_ids)
        else:
            query_tensors = [histories[i].tokens for i in active_histories]

        response_tensors,past_key_values,past_attention_masks,past_input_ids, truncated = self._generate_batched(query_tensors,combined_past_key_values=combined_past_key_values,combined_past_attention_masks=combined_past_attention_masks, combined_past_input_ids=combined_past_input_ids)
        if not truncated:
            response_texts = self.tokenizer.batch_decode(response_tensors)
            for i, response_text, response_tensor in zip(active_histories, response_texts, response_tensors):
                history = histories[i]
                if not history.completed:
                    history.append_segment(response_text, response_tensor, system=False)
        else:
            for history in histories:
                if not history.completed:
                    #Adds an eos token, so that we always end on a non-system segment
                    history.append_segment(self.tokenizer.eos_token, torch.tensor([self.tokenizer.eos_token_id]).to(self.current_device), system=False)
                    history.complete(truncated=True)

        return histories,past_key_values,past_attention_masks,past_input_ids, active_histories

    def tasks_end_check(self, histories, model_turn=True):
        """
        Check if the current generation sequences have finished.
        """

        for history in histories:
            if not history.completed:
                truncated, ended = self.task_end_check(history, model_turn=model_turn)
                if ended:
                    history.complete(truncated=truncated)
        return histories

    def task_end_check(self, history, model_turn=True):
        """
        Check if the current generation sequence has finished.
        """
        truncated = False
        ended = False
        if history.completed:
            return truncated, ended
        if self.max_length is not None and len(history.tokens) > self.max_length:
            truncated = True
            ended = True
        elif self.tokenizer.eos_token in history.text:
            ended = True
        elif model_turn and not (
            (self.request_token in history.last_text_segment and self.call_token in history.last_text_segment)
            or self.submit_token in history.last_text_segment
        ):
            ended = True
        elif self.submit_token in history.last_text_segment:
            ended = True
        return truncated, ended

    #builds the cache for the current batch
    def _get_batched_cache(self,start_index, end_index, combined_past_key_values, combined_attention_masks, combined_input_ids):
        """
        Extract (batch) cache for current batch
        start_index (int): start index of current batch
        end_index (int): end index of current batch (points to first element not in batch)
        combined_past_key_values (tuple[tuple[torch.Tensor]]) : The combined (unbatched) cache in legacy format from the last generation
        combined_past_attention_masks (torch.Tensor): The combined (unbatched) attention masks from the last generation
        combined_past_input_ids (torch.Tensor): The combined (unbatched) input ids from the last generation
        """
        current_cache = []
        for layer_id, layer in enumerate(combined_past_key_values):
            keys,values = layer
            new_keys = keys[start_index:end_index]
            new_values = values[start_index:end_index]
            current_cache.append((new_keys,new_values))
        current_cache = tuple(current_cache)
        return DynamicCache().from_legacy_cache(current_cache), combined_attention_masks[start_index:end_index], combined_input_ids[start_index:end_index]


    #TODO make batch_size changeable
    def _generate_batched(
        self,
        query_tensors,
        batch_size: int = 16,
        pad_to_multiple_of: Optional[int] = None,
        combined_past_key_values=None,
        combined_past_attention_masks=None,
        combined_past_input_ids = None
    ):
        """
        Generate responses for a list of query tensors.
        Either all of combined_past_key_values, combined_past_attention_masks, combined_past_input_ids are provided or all are None.
        Args:
            query_tensors (list[torch.Tensor]): A list of query tensors to generate responses for.
            batch_size (int): The batch size to use for generation.
            pad_to_multiple_of (int): The padding length to use for generation.
            combined_past_key_values (Optional[tuple[tuple[torch.Tensor]]]) : The combined (unbatched) cache in legacy format from the last generation
            combined_past_attention_masks (Optional[torch.Tensor]): The combined (unbatched) attention masks from the last generation
            combined_past_input_ids (Optional[torch.Tensor]): The combined (unbatched) input ids from the last generation
        """
        outputs = []
        padding_side_default = self.tokenizer.padding_side
        if not self.is_encoder_decoder:
            self.tokenizer.padding_side = "left"

        new_past_key_values = []
        new_past_attention_masks = []
        new_past_input_ids = []
        # in case we have fewer examples than bs
        batch_size = min(len(query_tensors), batch_size)
        for batch_index,i in enumerate(range(0, len(query_tensors), batch_size)):
            # prevent overflow if query tensors are not even multiple of bs
            end_index = min(len(query_tensors), i + batch_size)
            batch = query_tensors[i:end_index]
            batch_mask = [torch.ones_like(element) for element in batch]
            past_key_values, past_attention_masks, past_input_ids = (None,None,None)
            if combined_past_key_values is not None:
                past_key_values, past_attention_masks, past_input_ids = self._get_batched_cache(i,end_index,combined_past_key_values,combined_past_attention_masks,combined_past_input_ids)
            inputs = {"input_ids": batch, "attention_mask": batch_mask}

            padded_inputs = self.tokenizer.pad(
                inputs,
                padding=True,
                max_length=None,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors="pt",
            ).to(self.current_device)
            input_attention_mask = padded_inputs["attention_mask"].clone()
            stopping_criteria = StringStoppingCriteria([self.call_token, self.submit_token], self.tokenizer)

            self.generation_kwargs["stopping_criteria"] = StoppingCriteriaList([stopping_criteria])
            self.generation_kwargs["use_cache"] = True
            self.generation_kwargs["return_dict_in_generate"] = True
            #handle caching
            self.generation_kwargs["past_key_values"] = past_key_values if past_key_values is not None else DynamicCache()
            if past_attention_masks is not None:
                padded_inputs["attention_mask"] = torch.concatenate([past_attention_masks,padded_inputs["attention_mask"]],dim=1)
            if past_input_ids is not None:
                padded_inputs["input_ids"] = torch.concatenate([past_input_ids,padded_inputs["input_ids"]],dim=1)

            if self.max_length is not None and padded_inputs["input_ids"].shape[-1]>self.max_length:
                return None, None, None,None, True

            generations = extract_model_from_parallel(self.model).generate(**padded_inputs, **self.generation_kwargs)
            new_past_key_values.append(generations.past_key_values)

            past_attention_mask = torch.ones_like(generations.sequences)
            #Don't attend to generated padding or eos tokens
            past_attention_mask[torch.logical_or(generations.sequences==self.tokenizer.eos_token_id, generations.sequences==self.tokenizer.pad_token_id)] = 0
            past_attention_mask[:,:input_attention_mask.shape[1]] = input_attention_mask

            generations = generations.sequences

            new_past_input_ids.append(generations)
            for generation, mask, generated_tokens, new_attention_mask in zip(
                generations, padded_inputs["attention_mask"], stopping_criteria.generated_tokens,past_attention_mask
            ):
                if not self.is_encoder_decoder:
                    output = generation[(1 - mask).sum() :]  # remove padding
                    padding_removed_past_attention_mask = new_attention_mask[(1 - mask).sum() :]
                else:
                    output = generation
                    padding_removed_past_attention_mask = new_attention_mask

                if not self.is_encoder_decoder:
                    output = output[(mask).sum() :]  # remove prompt
                    generated_tokens_attention_mask = padding_removed_past_attention_mask[(mask).sum() :]

                # remove chunk generated after stopping criteria in batch mode
                outputs.append(output[:generated_tokens])
                #Do not attend to tokens that were generated after <call> or <submit>
                generated_tokens_attention_mask[generated_tokens:]=0
            new_past_attention_masks.append(past_attention_mask)
        self.tokenizer.padding_side = padding_side_default
        return outputs, new_past_key_values, new_past_attention_masks,new_past_input_ids, False
