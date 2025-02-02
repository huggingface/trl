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

import copy
import re
from typing import Optional

import torch
from accelerate.utils import extract_model_from_parallel
from transformers import DynamicCache, StoppingCriteria, StoppingCriteriaList

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
            self.generated_tokens = [0 for _ in range(input_ids.shape[0])]
            self.start_length = input_ids.shape[-1] - 1
            self.first_call = False
            self.last_done = [False for _ in range(input_ids.shape[0])]
        decoded_generations = self.tokenizer.batch_decode(input_ids[:, self.start_length :])
        done = []

        for i, decoded_generation in enumerate(decoded_generations):
            sequence_complete = (
                any(stop_string in decoded_generation for stop_string in self.stop_strings)
                or self.tokenizer.eos_token_id in input_ids[i, self.start_length :]
            )
            done.append(sequence_complete)
            # we still consider the last generated token to be valid
            if not self.last_done[i]:
                self.generated_tokens[i] += 1

        self.last_done = done

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
        self.logits = []
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

    def append_segment(self, text, tokens, system=True, logits=None):
        """
        Append a new segment to the history.

        Args:
            text (`str`): The text of the new segment.
            tokens (`torch.LongTensor`): The tokens of the new segment.
            system (`bool`, *optional*): Whether the new segment is a system or user segment.
            logits (`torch.Tensor`, *optional*): The logits for a non-system segment.
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
            if logits is not None:
                self.logits.append(logits)
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
                "The `rich` library is required to display text with formatting. Install it using `pip install rich`."
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
        use_cache=False,
        save_logits=False,
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
            use_cache (bool): Whether to cache past_key_values between segments. When using caching, [`TextEnvironment`] is not suited for training use, i.e. backpropagation through the generated graph. Use with Trainers is of course possible. Furthermore, caching requires, that there be no calculation dependencies between examples at inference time. When using `BatchNorm`, the model should thus be in eval mode.
            save_logits (bool): Whether to save logits in the returned histories. Mainly intended to help the user test caching for their use case. Backpropagation through logits is not supported.
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
        self.save_logits = save_logits

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

        past_key_values, past_attention_masks, past_input_ids, last_active_histories = (None, None, None, None)

        while any(not history.completed for history in histories) and turns < self.max_turns:
            if self.use_cache:
                histories, past_key_values, past_attention_masks, past_input_ids, last_active_histories = (
                    self.generate(
                        histories, past_key_values, past_attention_masks, past_input_ids, last_active_histories
                    )
                )
            else:
                # Discard cache
                histories, _, _, _, _ = self.generate(
                    histories, past_key_values, past_attention_masks, past_input_ids, last_active_histories
                )
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
            self.tokenizer(response + self.response_token, return_tensors="pt", add_special_tokens=False)
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

    def _next_input(self, history):
        return history.last_token_segment if not history.completed else torch.tensor([])

    def _combine_cache(self, example_mask, past_key_values, past_attention_masks, past_input_ids):
        """
        combines all caches in order to exclude completed histories from further generation

        Args:
            example_mask (list[bool]): mask indicating for each example, whether it is supposed to remain or not
            past_key_values (tuple[tuple[torch.Tensor]]) : Batched list of caches (in legacy format) from the last generation
            past_attention_masks (list[torch.Tensor]): Batched list of attention masks from the last generation
            past_input_ids (list[torch.Tensor]): Batched list of input ids from the last generation
        """
        max_sequence_length = max([attention_mask.shape[1] for attention_mask in past_attention_masks])

        combined_cache = []
        for layer_id in range(len(past_key_values[0])):
            combined_layer = None
            example_mask_offset = 0
            for cache in past_key_values:
                layer = cache[layer_id]
                num_examples = len(layer[0])
                extracted_keys = layer[0][example_mask[example_mask_offset : example_mask_offset + num_examples]]
                extracted_values = layer[1][example_mask[example_mask_offset : example_mask_offset + num_examples]]

                # pad to max_sequence_length -1
                new_keys = torch.zeros(
                    (
                        extracted_keys.shape[0],
                        extracted_keys.shape[1],
                        max_sequence_length - 1,
                        extracted_keys.shape[3],
                    )
                ).to(self.current_device)

                if extracted_keys.shape[2] != extracted_values.shape[2]:
                    raise Exception("Cache format incompatible")
                if extracted_keys.shape[2] > max_sequence_length - 1:
                    raise Exception("Cache sequence length is too large")
                start_position = max_sequence_length - 1 - extracted_keys.shape[2]
                if start_position < 0:
                    raise Exception("start position incorrect")
                new_values = torch.zeros_like(new_keys).to(self.current_device)
                new_keys[:, :, start_position:, :] = extracted_keys
                new_values[:, :, start_position:, :] = extracted_values

                if combined_layer is None:
                    combined_layer = (new_keys, new_values)
                else:
                    other_new_keys, other_new_values = combined_layer
                    combined_layer = (
                        torch.concat([other_new_keys, new_keys], dim=0),
                        torch.concat([other_new_values, new_values], dim=0),
                    )
                example_mask_offset += num_examples
            if example_mask_offset != len(example_mask):
                raise Exception("example_mask size and cache size are different")
            combined_cache.append(combined_layer)
        combined_cache = tuple(combined_cache)

        padded_attentions_masks = []
        padded_past_input_ids = []
        for attention_mask, input_ids in zip(past_attention_masks, past_input_ids):
            if attention_mask.shape[1] != input_ids.shape[1]:
                raise Exception("Cache format incompatible")
            start_position = max_sequence_length - attention_mask.shape[1]
            if start_position < 0:
                raise Exception("start position incorrect")
            padded_attention_mask = torch.zeros(
                (attention_mask.shape[0], max_sequence_length), dtype=attention_mask.dtype
            ).to(self.current_device)
            padded_attention_mask[:, start_position:] = attention_mask
            padded_attentions_masks.append(padded_attention_mask)

            padded_input_ids = torch.full(
                (input_ids.shape[0], max_sequence_length), self.tokenizer.pad_token_id, dtype=input_ids.dtype
            ).to(self.current_device)
            padded_input_ids[:, start_position:] = input_ids
            padded_past_input_ids.append(padded_input_ids)

        combined_attention_masks = torch.concat(padded_attentions_masks, dim=0)
        if combined_attention_masks.shape[0] != len(example_mask):
            raise Exception("example_mask and attention_masks have varying example counts")
        combined_attention_masks = combined_attention_masks[example_mask]
        combined_input_ids = torch.concat(padded_past_input_ids, dim=0)
        if combined_input_ids.shape[0] != len(example_mask):
            raise Exception("example_mask and input ids have varying example counts")
        combined_input_ids = combined_input_ids[example_mask]
        return combined_cache, combined_attention_masks, combined_input_ids

    def _same_is_none(self, *values):
        """For input validation
        Args:
            values: list[object]: A list of values to test for having the same return value for `is None`
        """
        expected_is_none = values[0] is None
        for value in values[1:]:
            if (value is None) != expected_is_none:
                return False
        return True

    def generate(
        self,
        histories,
        past_key_values=None,
        past_attention_masks=None,
        past_input_ids=None,
        last_active_histories=None,
    ):
        """
        Generate responses for a list of histories.
        Either all of past_key_values, past_attention_masks, past_input_ids,last_active_histories are provided or all are None.
        Args:
            histories (list[TextHistory]): A complete list of the TextHistories
            past_key_values (Optional[tuple[tuple[torch.Tensor]]]): Batched list of caches in legacy format from the last generation
            past_attention_masks (Optional[list[torch.Tensor]]): Batched list of attention masks from the last generation
            past_input_ids (Optional[list[torch.Tensor]]): Batched list of input ids from the last generation
            last_active_histories (Optional[list[int]]): indices of histories for which generation took place during the last generation turn
        """
        if not self._same_is_none(past_key_values, past_attention_masks, past_input_ids, last_active_histories):
            raise Exception("Either all cache related inputs are supposed to be None or all are not None.")

        active_histories = [i for i in range(len(histories)) if not histories[i].completed]
        combined_past_key_values, combined_past_attention_masks, combined_past_input_ids = (None, None, None)

        if past_key_values is not None:
            query_tensors = [self._next_input(histories[i]) for i in active_histories]
            example_mask = [(not histories[i].completed) for i in last_active_histories]
            combined_past_key_values, combined_past_attention_masks, combined_past_input_ids = self._combine_cache(
                example_mask, past_key_values, past_attention_masks, past_input_ids
            )
        else:
            query_tensors = [histories[i].tokens for i in active_histories]

        response_tensors, past_key_values, past_attention_masks, past_input_ids, truncated, logits = (
            self._generate_batched(
                query_tensors,
                combined_past_key_values=combined_past_key_values,
                combined_past_attention_masks=combined_past_attention_masks,
                combined_past_input_ids=combined_past_input_ids,
                return_cache=self.use_cache,
                output_logits=self.save_logits,
            )
        )
        if not truncated:
            response_texts = self.tokenizer.batch_decode(response_tensors)
            for i, response_text, response_tensor, j in zip(
                active_histories, response_texts, response_tensors, range(len(active_histories))
            ):
                history = histories[i]
                if not history.completed:
                    history.append_segment(
                        response_text, response_tensor, system=False, logits=(logits[j] if self.save_logits else None)
                    )
        else:
            for history in histories:
                if not history.completed:
                    history.complete(truncated=True)
            return histories, None, None, None, []  # invalidate cache

        return histories, past_key_values, past_attention_masks, past_input_ids, active_histories

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

    # builds the cache for the current batch
    def _get_batched_cache(
        self, start_index, end_index, combined_past_key_values, combined_attention_masks, combined_input_ids
    ):
        """
        Extract (batch) cache, attention_mask and input_ids for current batch
        Args:
            start_index (int): start index of current batch
            end_index (int): end index of current batch (points to first element not in batch)
            combined_past_key_values (tuple[tuple[torch.Tensor]]) : The combined (unbatched) cache in legacy format from the last generation
            combined_past_attention_masks (torch.Tensor): The combined (unbatched) attention masks from the last generation
            combined_past_input_ids (torch.Tensor): The combined (unbatched) input ids from the last generation
        """
        current_cache = []
        for layer in combined_past_key_values:
            keys, values = layer
            new_keys = keys[start_index:end_index]
            new_values = values[start_index:end_index]
            current_cache.append((new_keys, new_values))
        current_cache = tuple(current_cache)
        return (
            current_cache,
            combined_attention_masks[start_index:end_index],
            combined_input_ids[start_index:end_index],
        )

    def _extract_generation(self, sequence, mask):
        """Remove padding and prompt based on the attention mask to extract generated tokens
        Args:
            sequence (torch.Tensor): A sequence with length corresponding to input sequence length + generation sequence length
            mask (torch.Tensor): The input attention mask
        """
        if not self.is_encoder_decoder:
            # remove padding
            output = sequence[(1 - mask).sum() :]
        else:
            output = sequence

        if not self.is_encoder_decoder:
            # remove prompt
            output = output[(mask).sum() :]
        return output

    def _create_new_past_attention_mask(self, sequences, input_attention_mask, generated_tokens):
        """Creates the new past_input_ids and new past_attention_mask for a batch.
        Args:
            sequences (torch.Tensor): The sequences returned by model.generate(...)
            input_attention_mask (torch.Tensor): The attention mask that was input into model.generate(...)
            generated_tokens (list[int]): The number of valid tokens generated for each history in the batch
        """
        new_past_attention_mask = torch.ones_like(sequences)
        new_past_attention_mask[:, : input_attention_mask.shape[1]] = input_attention_mask
        # copy for in-place modification
        for mask, num_generated_tokens, new_attention_mask in zip(
            input_attention_mask,
            generated_tokens,
            new_past_attention_mask,
        ):
            extracted_past_attention_mask = self._extract_generation(new_attention_mask, mask)
            # Do not attend to invalid tokens that were generated after <call> or <submit>
            extracted_past_attention_mask[num_generated_tokens:] = 0
        return new_past_attention_mask

    # TODO make batch_size changeable
    def _generate_batched(
        self,
        query_tensors,
        batch_size: int = 16,
        pad_to_multiple_of: Optional[int] = None,
        combined_past_key_values=None,
        combined_past_attention_masks=None,
        combined_past_input_ids=None,
        output_logits=False,
        return_cache=False,
    ):
        """
        Generate responses for a list of query tensors.
        Either all of combined_past_key_values, combined_past_attention_masks, combined_past_input_ids are provided or all are None.
        Args:
            query_tensors (list[torch.Tensor]): A list of non-empty query tensors to generate responses for.
            batch_size (int): The batch size to use for generation.
            pad_to_multiple_of (int): The padding length to use for generation.
            combined_past_key_values (Optional[tuple[tuple[torch.Tensor]]]) : The combined (unbatched) cache in legacy format from the last generation
            combined_past_attention_masks (Optional[torch.Tensor]): The combined (unbatched) attention masks from the last generation
            combined_past_input_ids (Optional[torch.Tensor]): The combined (unbatched) input ids from the last generation
        """
        if not self._same_is_none(combined_past_key_values, combined_past_attention_masks, combined_past_input_ids):
            raise Exception("Either all cache related inputs are supposed to be None or all are not None.g")

        caching_enabled = return_cache or (combined_past_key_values is not None)
        # Ensures, that the next token is never conditioned on a padding token. This should never be a problem, as empty system prompts are not particularly useful and between segments there is always a response token.
        for query in query_tensors:
            if len(query) == 0:
                raise Exception("Cannot input empty query")
        outputs = []
        padding_side_default = self.tokenizer.padding_side
        if not self.is_encoder_decoder:
            self.tokenizer.padding_side = "left"

        if return_cache:
            new_past_key_values, new_past_attention_masks, new_past_input_ids = ([], [], [])
        else:
            new_past_key_values, new_past_attention_masks, new_past_input_ids = (None, None, None)

        if output_logits:
            all_logits = []
        else:
            all_logits = None

        # in case we have fewer examples than bs
        batch_size = min(len(query_tensors), batch_size)
        for i in range(0, len(query_tensors), batch_size):
            # prevent overflow if query tensors are not even multiple of bs
            end_index = min(len(query_tensors), i + batch_size)
            past_key_values, past_attention_masks, past_input_ids = (None, None, None)
            if combined_past_key_values is not None:
                past_key_values, past_attention_masks, past_input_ids = self._get_batched_cache(
                    i, end_index, combined_past_key_values, combined_past_attention_masks, combined_past_input_ids
                )

            query_batch = query_tensors[i:end_index]
            mask = [torch.ones_like(element) for element in query_batch]
            inputs = {"input_ids": query_batch, "attention_mask": mask}
            padded_inputs = self.tokenizer.pad(
                inputs,
                padding=True,
                max_length=None,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors="pt",
            ).to(self.current_device)
            stopping_criteria = StringStoppingCriteria([self.call_token, self.submit_token], self.tokenizer)

            generation_kwargs = copy.deepcopy(self.generation_kwargs)

            generation_kwargs["stopping_criteria"] = StoppingCriteriaList([stopping_criteria])
            generation_kwargs["return_dict_in_generate"] = True

            if output_logits:
                generation_kwargs["output_logits"] = True

            if caching_enabled:
                generation_kwargs["use_cache"] = True
                generation_kwargs["return_legacy_cache"] = True
            if past_attention_masks is not None:
                padded_inputs["attention_mask"] = torch.concatenate(
                    [past_attention_masks, padded_inputs["attention_mask"]], dim=1
                )
            if past_input_ids is not None:
                padded_inputs["input_ids"] = torch.concatenate([past_input_ids, padded_inputs["input_ids"]], dim=1)

            if self.max_length is not None and padded_inputs["input_ids"].shape[-1] > self.max_length:
                return None, None, None, None, True

            extracted_model = extract_model_from_parallel(self.model)
            if caching_enabled and extracted_model.pretrained_model._supports_cache_class:
                generation_kwargs["past_key_values"] = (
                    DynamicCache().from_legacy_cache(past_key_values)
                    if past_key_values is not None
                    else DynamicCache()
                )
            elif caching_enabled:
                generation_kwargs["past_key_values"] = past_key_values

            cloned_attention_mask = padded_inputs["attention_mask"].clone()
            generations = extracted_model.generate(**padded_inputs, **generation_kwargs)

            if output_logits:
                logits = generations.logits
            sequences = generations.sequences
            for generation, mask, num_generated_tokens in zip(
                sequences, cloned_attention_mask, stopping_criteria.generated_tokens
            ):
                output = self._extract_generation(generation, mask)
                # remove chunk generated after stopping criteria in batch mode
                generated_tokens = output[:num_generated_tokens]
                if len(generated_tokens) < 1:
                    raise Exception("Generation failed to produce any valid tokens")

                outputs.append(generated_tokens)

            if return_cache:
                if generations.past_key_values[0][0].shape[2] != generations.sequences.shape[1] - 1:
                    raise Exception("Cache should not contain keys and values for last generated token")
                new_past_key_values.append(generations.past_key_values)
                new_past_attention_mask = self._create_new_past_attention_mask(
                    sequences, cloned_attention_mask, stopping_criteria.generated_tokens
                )
                new_past_attention_masks.append(new_past_attention_mask)
                new_past_input_ids.append(sequences.clone())

            if output_logits:
                for i, num_generated_tokens in enumerate(stopping_criteria.generated_tokens):
                    relevant_logits = [batched_logits[i] for batched_logits in logits[:num_generated_tokens]]
                    all_logits.append(torch.stack(relevant_logits, dim=0).detach().clone())

        self.tokenizer.padding_side = padding_side_default

        return outputs, new_past_key_values, new_past_attention_masks, new_past_input_ids, False, all_logits
