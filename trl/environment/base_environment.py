import re
import warnings

import torch
from accelerate.utils import extract_model_from_parallel
from transformers import StoppingCriteria, StoppingCriteriaList


is_rich_available = True
try:
    from rich import print
    from rich.text import Text
except ImportError:
    is_rich_available = False


class StringStoppingCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generations in the batch are completed."""

    def __init__(self, stop_strings, tokenizer):
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        self.first_call = True

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the stop strings."""
        if self.first_call:
            self.generated_tokens = [1 for _ in range(input_ids.shape[0])]
            self.start_length = input_ids.shape[-1] - 1
            self.first_call = False
        decoded_generations = self.tokenizer.batch_decode(input_ids[:, self.start_length :])
        done = []

        for i, decoded_generation in enumerate(decoded_generations):
            sequence_complete = any([stop_string in decoded_generation for stop_string in self.stop_strings])
            done.append(sequence_complete)
            if not sequence_complete:
                self.generated_tokens[i] += 1

        if all(done):
            self.first_call = True

        return all(done)


class TextHistory:
    def __init__(self, text, tokens, system=True):
        self.system_spans = []
        self.text_spans = []
        self.token_spans = []
        self.token_masks = torch.tensor([], dtype=torch.long).to(tokens.device)
        self.text = ""
        self.tokens = torch.tensor([], dtype=torch.long).to(tokens.device)
        self.completed = False
        self.truncated = False
        self.reward = 0.0

        self.append_segment(text, tokens, system=system)

    def append_segment(self, text, tokens, system=True):
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
        self.completed = True
        self.truncated = truncated

    @property
    def last_text_segment(self):
        start, end = self.text_spans[-1]
        return self.text[start:end]

    def split_query_response_tokens(self):
        split_index = self.token_spans[0][1]
        query = self.tokens[:split_index]
        response = self.tokens[split_index:]
        mask = self.token_masks[split_index:]

        return query, response, mask

    def show_text(self):
        if not is_rich_available:
            warnings.warn("install rich to display text")
            return

        text = Text(self.text)
        text.stylize("black on grey85", self.text_spans[0][0], self.text_spans[1][0])
        for i, (start, end) in enumerate(self.text_spans[1:]):
            if self.system_spans[i]:
                color = "cyan3"
            else:
                color = "deep_sky_blue1"
            text.stylize(f"black on {color}", start, end)
        print(text)
        text = Text(f"Reward: {self.reward}")
        text.stylize("black on plum1", self.text_spans[0][0], self.text_spans[0][1])
        print(text)

    def show_tokens(self, tokenizer):
        if not is_rich_available:
            warnings.warn("install rich to display text")
            return

        text = Text()
        prompt_end = self.token_spans[0][1]
        for i, (token, mask) in enumerate(zip(self.tokens, self.token_masks)):
            if i < prompt_end:
                text.append(tokenizer.convert_ids_to_tokens(token.item()), style="black on grey85")
                text.append(" ")
            elif mask == 1:
                text.append(tokenizer.convert_ids_to_tokens(token.item()), style="black on deep_sky_blue1")
                text.append(" ")
            else:
                text.append(tokenizer.convert_ids_to_tokens(token.item()), style="black on cyan3")
                text.append(" ")
        text.append(f"\n\nReward: {self.reward}", style="black on plum1")
        print(text)


class TextEnvironment:
    """
    The TextEnvironment enables interaction of a LLM with an environment using tools.
    """

    def __init__(
        self, model, tokenizer, tools, reward_fn, prompt, max_turns=4, max_tool_reponse=100, generation_kwargs=None
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
            generation_kwargs (Optional[dict]): A dictionary of keyword arguments to pass to the model's generate method.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        if isinstance(tools, dict):
            self.tools = tools
        else:
            self.tools = dict([(tool.__class__.__name__, tool) for tool in tools])
        self.reward_fn = reward_fn
        self.max_length = None
        self.request_token = "<request>"
        self.call_token = "<call>"
        self.response_token = "<response>"
        self.submit_token = "<submit>"
        self.max_turns = max_turns
        self.max_tool_response = max_tool_reponse

        if generation_kwargs is None:
            self.generation_kwargs = dict()
        else:
            self.generation_kwargs = generation_kwargs

        self.is_encoder_decoder = hasattr(self.model, "is_encoder_decoder")
        self.current_device = extract_model_from_parallel(self.model).pretrained_model.device

    def run(self, tasks, **rewards_kwargs):
        """
        Run the environment on a list of tasks.

        Args:
            tasks (list[str]): A list of tasks to run the model in the environment on.
        """
        turns = 0

        queries = [self.prompt + task for task in tasks]
        queries_tokens = [
            self.tokenizer(query, return_tensors="pt").input_ids[0].to(self.model.pretrained_model.device)
            for query in queries
        ]

        histories = [TextHistory(q, qt, system=True) for q, qt in zip(queries, queries_tokens)]

        while any([not history.completed for history in histories]):
            self.generate(histories)
            # TODO: make this parallel rather than for-loop
            for i in range(len(histories)):
                self.step(histories[i])
            turns += 1
            if turns == self.max_turns:
                break

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
            return

        try:
            tool, query = self.parse_tool_call(history.last_text_segment)
            if tool not in self.tools:
                response = f"Uknown tool {tool}."
            try:
                response = self.tools[tool](query)
            except Exception as error:
                response = f"Tool error: {str(error)}"
        except Exception as error:
            response = f"Invalid tool call: {str(error)}"

        if len(response) > self.max_tool_response:
            response = response[: (self.max_tool_response - 3)] + "..."

        history.append_segment(
            response + self.response_token,
            self.tokenizer(response + self.response_token, return_tensors="pt")
            .input_ids[0]
            .to(self.model.pretrained_model.device),
            system=True,
        )

    def parse_tool_call(self, text):
        """
        Parse request string. Expected format: <request><tool_name>query<call>
        """
        result = re.search(f"(?<={self.request_token}).*?(?={self.call_token})", text, re.DOTALL)
        extracted_text = result.group()

        tool = re.search(r"<(.*?)>", extracted_text).group(1)
        query = ">".join(extracted_text.split(">")[1:])

        return tool, query

    def compute_reward(self, histories, **reward_kwargs):
        """
        Compute the reward for a list of histories.
        """
        rewards = self.reward_fn([history.last_text_segment for history in histories], **reward_kwargs)
        for history, reward in zip(histories, rewards):
            history.reward = torch.tensor(reward)
        return histories

    def generate(self, histories):
        """
        Generate responses for a list of histories.
        """
        active_histories = [i for i, history in enumerate(histories) if not history.completed]

        query_tensors = [histories[i].tokens for i in active_histories]
        response_tensors = self._generate_batched(query_tensors)
        response_texts = self.tokenizer.batch_decode(response_tensors)

        for i, response_text, response_tensor in zip(active_histories, response_texts, response_tensors):
            histories[i].append_segment(response_text, response_tensor, system=False)

    def task_end_check(self, history):
        """
        Check if the current generation sequence has finished.
        """
        truncated = False
        ended = False
        if history.completed:
            return truncated, ended
        if self.max_length is not None and len(history.text) > self.max_length:
            truncated = True
            ended = True
        elif self.tokenizer.eos_token in history.text:
            ended = True
        elif not (
            (self.request_token in history.last_text_segment and self.call_token in history.last_text_segment)
            or self.submit_token in history.last_text_segment
        ):
            ended = True
        elif self.submit_token in history.last_text_segment:
            ended = True
        return truncated, ended

    def _generate_batched(
        self,
        query_tensors,
        batch_size: int = 16,
        pad_to_multiple_of: int = None,
    ):
        """
        Generate responses for a list of query tensors.

        args:
            query_tensors (list[torch.Tensor]): A list of query tensors to generate responses for.
            batch_size (int): The batch size to use for generation.
            pad_to_multiple_of (int): The padding length to use for generation.
        """
        outputs = []
        padding_side_default = self.tokenizer.padding_side
        if not self.is_encoder_decoder:
            self.tokenizer.padding_side = "left"

        # in case we have fewer examples than bs
        batch_size = min(len(query_tensors), batch_size)

        for i in range(0, len(query_tensors), batch_size):
            # prevent overflow if query tensors are not even multiple of bs
            end_index = min(len(query_tensors), i + batch_size)

            batch = query_tensors[i:end_index]
            batch_mask = [torch.ones_like(element) for element in batch]
            inputs = {"input_ids": batch, "attention_mask": batch_mask}

            padded_inputs = self.tokenizer.pad(
                inputs,
                padding=True,
                max_length=None,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors="pt",
            ).to(self.current_device)

            stopping_criteria = StringStoppingCriteria([self.call_token, self.submit_token], self.tokenizer)

            self.generation_kwargs["stopping_criteria"] = StoppingCriteriaList([stopping_criteria])

            generations = extract_model_from_parallel(self.model).generate(**padded_inputs, **self.generation_kwargs)

            for generation, mask, generated_tokens in zip(
                generations, padded_inputs["attention_mask"], stopping_criteria.generated_tokens
            ):
                if not self.is_encoder_decoder:
                    output = generation[(1 - mask).sum() :]  # remove padding
                else:
                    output = generation

                if not self.is_encoder_decoder:
                    output = output[(mask).sum() :]  # remove prompt

                # remove chunk generated after stopping criteria in batch mode
                outputs.append(output[:generated_tokens])
        self.tokenizer.padding_side = padding_side_default
        return outputs
