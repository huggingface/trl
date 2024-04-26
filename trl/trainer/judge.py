from typing import List
import random

import os
import random
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List

import numpy as np
from datasets import load_dataset

from ..import_utils import is_pairm_available

# import anthropic
# import vertexai
from accelerate import Accelerator
# from openai import BadRequestError, OpenAI
# from vertexai.preview.generative_models import GenerationConfig, GenerativeModel

if is_pairm_available():
    import llm_blender


LMSYS_SYSTEM_TEMPLATE = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie."""


LMSYS_PROMPT_TEMPLATE = """"[User Question]\n{prompt}\n\n[The Start of Assistant A's Answer]\n{response1}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{response2}\n[The End of Assistant B's Answer]"""

class MockJudge:
    def judge_batch(self, prompts: List[str], completion_pairs: List[List[str]]) -> List[int]:
        results = []
        for prompt, completion_pair in zip(prompts, completion_pairs):
            results.append(random.choice([0, 1]))

        return results

class FutureAnnotator:
    def __init__(self, max_workers=8):
        self.thread_pool_executor = ThreadPoolExecutor(max_workers=max_workers)

    def __del__(self) -> None:
        self.thread_pool_executor.shutdown()

    def get_reply(self, content: str) -> str:
        raise NotImplementedError

    def judge(
        self, prompt: str, completion_pair: List[str], shuffle_order: bool, max_tokens: int = 3, max_tries: int = 5
    ) -> int:
        if max_tries == 0:
            print("Max retries reached")
            return random.choice([0, 1])

        shuffle_index = 0 if not shuffle_order else random.choice([0, 1])

        content = HELPFULNESS_PROMPT.format(
            prompt=prompt, response1=completion_pair[shuffle_index], response2=completion_pair[1 - shuffle_index]
        )
        reply = self.get_reply(content)
        reply = reply.strip()

        # answer_index = reply.find("[[")+2
        # if answer_index == -1:
        #     return random.choice([0, 1])
        # if reply[answer_index] == "A":
        #     return 0
        # elif reply[answer_index] == "B":
        #     return 1
        # else:
        #     return random.choice([0, 1])

        # First answer
        if reply in [
            "1",
            "Option 1",
            "Summary 1",
            "Response 1",
            "The first response",
            "Answer 1",
        ]:
            return shuffle_index
        # Second answer
        elif reply in [
            "2",
            "Option 2",
            "Summary 2",
            "Response 2",
            "The second response",
            "Answer 2",
        ]:
            return 1 - shuffle_index
        # Ties
        elif reply in [
            "Both responses are",
            "Both Response",
            "Neither response",
            "Neither",
            "Neither response is",
            "Neither response addresses",
            "Neither response addresses",
            "Neither response correctly",
            "Both responses provided",
            "Both responses provide",
            "Neither response provides",
            "The two responses",
            "Both responses here",
        ]:
            return random.choice([0, 1])
        # Unknown reply
        else:
            print("Error: ", reply)
            return self.judge(prompt, completion_pair, max_tokens, shuffle_order, max_tries - 1)

    def judge_single(self, prompt: str, completion_pair: List[str], shuffle_order: bool = True) -> Future:
        return self.thread_pool_executor.submit(self.judge, prompt, completion_pair, shuffle_order)

    def judge_batch(
        self, prompts: List[str], completion_pairs: List[List[str]], shuffle_order: bool = True
    ) -> List[int]:
        futures = []
        for prompt, completion_pair in zip(prompts, completion_pairs):
            future = self.judge_single(prompt, completion_pair, shuffle_order=shuffle_order)
            futures.append(future)

        results = [f.result() for f in futures]

        return results


class OpenAIAnnotator(FutureAnnotator):
    def __init__(self, max_workers=8, model_name="gpt-4-turbo-preview"):
        super().__init__(max_workers=max_workers)
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=5)
        self.model_name = model_name

    def get_reply(self, content: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    # {"role": "system", "content": LMSYS_SYSTEM_TEMPLATE},
                    {"role": "user", "content": content}
                ],
                max_tokens=3,
            )
            reply = response.choices[0].message.content
            return reply
        except BadRequestError as e:
            print("BadRequestError", e)
            print("Content: ", content)
            return random.choice(["0", "1"])


class GeminiAnnotator(FutureAnnotator):
    def __init__(self, model_name="gemini-1.0-pro"):
        super().__init__()
        vertexai.init(project="ed-gemini", location="us-central1")
        self.model_name = model_name
        generation_config = GenerationConfig(
            # temperature=0.1,
            # top_p=0.95,
            # top_k=20,
            # candidate_count=1,
            max_output_tokens=3,
            # stop_sequences=["STOP!"],
        )
        self.model = GenerativeModel(model_name, generation_config=generation_config)

    def get_reply(self, content: str):
        # response_validation=False as otherwise it raises an exception when max_tokens is reached
        chat = self.model.start_chat(response_validation=False)
        response = chat.send_message(content)
        return response.text


class ClaudeAnnotator(FutureAnnotator):
    def __init__(self, model_name="claude-3-opus-20240229"):
        super().__init__()
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model_name = model_name

    def get_reply(self, content: str) -> str:
        message = self.client.messages.create(
            model="claude-3-opus-20240229", max_tokens=3, messages=[{"role": "user", "content": content}]
        )
        reply = message.content[0].text
        return reply


class PairRMAnnotator:
    def __init__(self):
        # from: https://huggingface.co/llm-blender/PairRM
        self.blender = llm_blender.Blender()
        self.blender.loadranker("llm-blender/PairRM", device=Accelerator().device)  # load PairRM

    def judge_batch(self, convs_a: List[List[Dict]], convs_b: List[List[Dict]]) -> List[int]:
        convs_a = [self._reformat_conv(c) for c in convs_a]
        convs_b = [self._reformat_conv(c) for c in convs_b]
        results = self.blender.compare_conversations(convs_a, convs_b)
        results_flipped = self.blender.compare_conversations(convs_b, convs_a)

        # average with random sample in the case of mismatch
        return [random.choice([1 - r0, r1 - 0]) for r0, r1 in zip(results, results_flipped)]

    @staticmethod
    def _reformat_conv(conv):
        return [{"role": c["role"].upper(), "content": c["content"]} for c in conv]  # user->USER, assistant->ASSISTANT


def evaluate_annotator(annotator: FutureAnnotator, dataset):
    batch = {
        "prompts": [],
        "completions": [],
    }
    for d in dataset:
        prompt = ""
        for message in d["messages"][:-1]:
            if message["role"] == "user":
                prompt += "Human: " + message["content"] + "\n"
            else:
                prompt += "Assistant: " + message["content"] + "\n"

        prompt += "Assistant: "
        completions = [
            d["rejected"][-1]["content"],
            d["chosen"][-1]["content"],
        ]
        batch["prompts"].append(prompt)
        batch["completions"].append(completions)

    results = annotator.judge_batch(batch["prompts"], batch["completions"], shuffle_order=True)

    peft = sum(results) / len(results)
    return peft


def evalate_annotators():
    # dataset = load_dataset("HuggingFaceH4/summarize_from_feedback", split="train_sft")
    # dataset = dataset.shuffle(seed=42).select(range(100))
    # dataset = load_dataset("HuggingFaceH4/hhh_alignment", "helpful", split="test")
    dataset = load_dataset("HuggingFaceH4/h4-anthropic-hh-rlhf-helpful-base")
    # print(dataset)
    dataset = dataset["train_prefs"].shuffle(seed=42).select(range(100))

    # def add_summary_to_prompt(example):
    #     prompt = example["prompt"]
    #     prompt += "\nSummary:\n"
    #     example["prompt"] = prompt
    #     return example

    # dataset.map(add_summary_to_prompt)

    annotators_dict = {
        "OpenAI-GPT4": OpenAIAnnotator(model_name="gpt-4-turbo-preview"),
        "OpenAI-GPT3.5": OpenAIAnnotator(model_name="gpt-3.5-turbo-0125"),
        # "ClaudeOpus": ClaudeAnnotator(model_name="gpt-3.5-turbo"),
        # "Gemini-1.0-Pro": GeminiAnnotator(model_name="gemini-1.0-pro"),
        # "Gemini-PaLM2": GeminiAnnotator(model_name="text-bison@002"),
    }
    results = defaultdict(list)

    for annotator in annotators_dict.keys():
        for eval in range(3):
            evaluator = annotators_dict[annotator]
            acc = evaluate_annotator(evaluator, dataset)
            results[annotator].append(acc)

    for annotator in results.keys():
        result = np.array(results[annotator])
        print(f"{annotator} | {result} {np.mean(result)} | {np.std(result)}")


def eval_positional_bias():
    # eval position ordering bias
    dataset = load_dataset("edbeeching/test_dataset", split="train")

    print(dataset)

    annotator = OpenAIAnnotator(model_name="gpt-3.5-turbo-0125", max_workers=8)

    for shuffle in [False, True]:
        accs = []
        for i in range(3):
            results = annotator.judge_batch(dataset["prompts"], dataset["completions"], shuffle_order=shuffle)  #
            acc = 1.0 - sum(results) / len(results)
            accs.append(acc)

        result = np.array(accs)
        print(f"{shuffle=} | {result} | {np.mean(result)} | {np.std(result)}")


if __name__ == "__main__":
    evalate_annotators()