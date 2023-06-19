import re

import torch
from accelerate.utils import extract_model_from_parallel
from rich.console import Console
from rich.text import Text
from transformers import StoppingCriteria, StoppingCriteriaList


class StringStoppingCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generations in the batch are completed."""

    def __init__(self, start_lengths, stop_strings, tokenizer):
        self.start_lengths = start_lengths
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the stop strings."""
        decoded_generations = self.tokenizer.batch_decode(input_ids)
        decoded_generations = [
            decoded_generation[start_length:]
            for start_length, decoded_generation in zip(self.start_lengths, decoded_generations)
        ]
        done = []
        for decoded_generation in decoded_generations:
            done.append(any([stop_string in decoded_generation for stop_string in self.stop_strings]))
        return all(done)


class TextHistory:
    def __init__(self, text, system=True):
        self.system_spans = []
        self.spans = []
        self.text = text
        self.completed = False
        self.truncated = False
        self.reward = 0.0

        if len(text) > 0:
            self.spans.append((0, len(text)))
            self.system_spans.append(system)

    def append(self, text, system=True):
        if len(text) == 0:
            raise ValueError("Can't append empty text to history.")
        original_text_length = len(self.text)
        self.text += text
        self.spans.append((original_text_length, len(self.text)))
        self.system_spans.append(system)

    def complete(self, truncated=False):
        self.completed = True
        self.truncated = truncated

    @property
    def last_text_segment(self):
        start, end = self.spans[-1]
        return self.text[start:end]

    def show(self):
        console = Console()
        text = Text(self.text)
        text.stylize("rgb(128,128,128)", self.spans[0][0], self.spans[1][0])
        for i, (start, end) in enumerate(self.spans[1:]):
            if self.system_spans[i]:
                color = "green"
            else:
                color = "blue"
            text.stylize(f"{color}", start, end)
        console.print(text)
        text = Text(f"Reward: {self.reward}")
        text.stylize("bold red", self.spans[0][0], self.spans[0][1])
        console.print(text)


class TextEnvironment:
    def __init__(self, model, tokenizer, tools, reward_fn, prompt, max_turns=4, generation_kwargs=None):
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

        if generation_kwargs is None:
            self.generation_kwargs = dict()
        else:
            self.generation_kwargs = generation_kwargs

        self.is_encoder_decoder = hasattr(self.model, "is_encoder_decoder")
        self.current_device = extract_model_from_parallel(self.model).pretrained_model.device

    def encode_history(self, history, is_system_prompt=False):
        encoded_prompt = self.tokenizer(history.last_text_segment, return_tensors="pt").to(self.current_device)
        input_ids = encoded_prompt["input_ids"]

        if is_system_prompt:
            attention_mask = torch.zeros_like(input_ids)
        else:
            attention_mask = torch.ones_like(input_ids)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def run(self, tasks):
        turns = 0
        histories = [TextHistory(self.prompt + task, system=True) for task in tasks]

        encoded_prompts = [
            self.tokenizer(self.prompt + task, return_tensors="pt").to(self.current_device) for task in tasks
        ]

        # queries
        histories_queries = [encoded_prompt["input_ids"].squeeze() for encoded_prompt in encoded_prompts]
        queries_masks = [encoded_prompt["attention_mask"].squeeze() for encoded_prompt in encoded_prompts]

        # respo,ses
        histories_responses = [[] for _ in range(len(tasks))]
        responses_masks = [[] for _ in range(len(tasks))]

        # rewards
        histories_rewards = []

        while any([not history.completed for history in histories]):
            histories = self.generate(histories)

            # TODO: make this parallel rather than for-loop
            for i in range(len(histories)):
                encoded_history = self.encode_history(histories[i])
                histories_responses[i].append(encoded_history["input_ids"].squeeze())
                responses_masks[i].append(encoded_history["attention_mask"].squeeze())

                histories[i] = self.step(histories[i])

                is_system_prompt = not histories[i].completed
                encoded_history = self.encode_history(histories[i], is_system_prompt=is_system_prompt)

                histories_responses[i].append(encoded_history["input_ids"].squeeze())
                responses_masks[i].append(encoded_history["attention_mask"].squeeze())

            turns += 1
            if turns == self.max_turns:
                break

        histories = self.compute_reward(histories)

        for i, history in enumerate(histories):
            histories_rewards.append(torch.Tensor([history.reward]))
            histories_responses[i] = torch.cat(histories_responses[i], dim=0).squeeze()
            responses_masks[i] = torch.cat(responses_masks[i], dim=0).squeeze()

        return (histories_queries, histories_responses), (responses_masks, queries_masks), histories_rewards, histories

    def step(self, history):
        history = self.task_end_check(history)
        if history.completed:
            return history

        try:
            tool, query = self.parse_tool_call(history.last_text_segment)
            response = self.tools[tool](query)
        except Exception as error:
            response = str(error)

        history.append(response + self.response_token, system=True)

        return history

    def parse_tool_call(self, text):
        """Parse request string. Expected format: <request><tool_name>query<call>"""
        result = re.search(f"(?<={self.request_token}).*?(?={self.call_token})", text)
        extracted_text = result.group()

        tool = re.search(r"<(.*?)>", extracted_text).group(1)
        query = ">".join(extracted_text.split(">")[1:])

        return tool, query

    def compute_reward(self, histories):
        for history in histories:
            history.reward = self.reward_fn(history.last_text_segment)
        return histories

    def generate(self, histories):
        active_histories = [i for i, history in enumerate(histories) if not history.completed]

        query_tensors = [
            self.tokenizer(histories[i].text, return_tensors="pt").input_ids.squeeze() for i in active_histories
        ]
        input_lengths = [len(histories[i].text) for i in active_histories]
        response_tensors = self._generate_batched(query_tensors, input_lengths)
        response_texts = self.tokenizer.batch_decode(response_tensors)

        for i, response_text in zip(active_histories, response_texts):
            histories[i].append(response_text, system=False)

        return histories

    def task_end_check(self, history):
        """Check if the current generation sequence has finished."""
        if history.completed:
            return history
        truncated = False
        ended = False
        if self.max_length is not None and len(history.text) > self.max_length:
            truncated = True
            ended = True
        elif self.tokenizer.eos_token in history.text:
            ended = True
        elif self.request_token not in history.last_text_segment:
            ended = True
        elif self.submit_token in history.last_text_segment:
            ended = True
        if ended:
            history.complete(truncated=truncated)
        return history

    def _generate_batched(
        self,
        query_tensors,
        input_lengths,
        batch_size: int = 4,
        pad_to_multiple_of: int = None,
    ):
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
            batch_input_lengths = input_lengths[i:end_index]
            batch_mask = [torch.ones_like(element) for element in batch]
            inputs = {"input_ids": batch, "attention_mask": batch_mask}

            padded_inputs = self.tokenizer.pad(
                inputs,
                padding=True,
                max_length=None,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors="pt",
            ).to(self.current_device)

            self.generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [StringStoppingCriteria(batch_input_lengths, [self.call_token, self.submit_token], self.tokenizer)]
            )

            generations = extract_model_from_parallel(self.model).generate(**padded_inputs, **self.generation_kwargs)

            for generation, mask in zip(generations, padded_inputs["attention_mask"]):
                if not self.is_encoder_decoder:
                    output = generation[(1 - mask).sum() :]  # remove padding
                else:
                    output = generation

                if not self.is_encoder_decoder:
                    output = output[(mask).sum() :]  # remove prompt
                outputs.append(output)

        self.tokenizer.padding_side = padding_side_default
        return outputs
