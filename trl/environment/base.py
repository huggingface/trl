import re

import torch
from accelerate.utils import extract_model_from_parallel
from rich import print
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

    def run(self, tasks):
        turns = 0

        queries = [self.prompt + task for task in tasks]
        queries_tokens = [
            self.tokenizer(query, return_tensors="pt").input_ids[0].to(self.model.pretrained_model.device)
            for query in queries
        ]

        histories = [TextHistory(q, qt, system=True) for q, qt in zip(queries, queries_tokens)]

        while any([not history.completed for history in histories]):
            histories = self.generate(histories)
            # TODO: make this parallel rather than for-loop
            for i in range(len(histories)):
                histories[i] = self.step(histories[i])
            turns += 1
            if turns == self.max_turns:
                break

        self.compute_reward(histories)

        # convert a list of (q, r, m) tuples to lists of all qs, rs, and ms respectively
        queries, responses, masks = map(list, zip(*[history.split_query_response_tokens() for history in histories]))

        rewards = [history.reward for history in histories]
        return queries, responses, masks, rewards, histories

    def step(self, history):
        history = self.task_end_check(history)
        if history.completed:
            return history

        try:
            tool, query = self.parse_tool_call(history.last_text_segment)
            response = self.tools[tool](query)
        except Exception as error:
            response = str(error)

        history.append_segment(
            response + self.response_token,
            self.tokenizer(response + self.response_token, return_tensors="pt")
            .input_ids[0]
            .to(self.model.pretrained_model.device),
            system=True,
        )

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

        query_tensors = [histories[i].tokens for i in active_histories]
        input_lengths = [len(histories[i].text) for i in active_histories]
        response_tensors = self._generate_batched(query_tensors, input_lengths)
        response_texts = self.tokenizer.batch_decode(response_tensors)

        for i, response_text, response_tensor in zip(active_histories, response_texts, response_tensors):
            histories[i].append_segment(response_text, response_tensor, system=False)

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
