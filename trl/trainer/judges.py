import os
import random
from concurrent.futures import Future, ThreadPoolExecutor
from typing import List

import llm_blender
from accelerate import Accelerator
from openai import BadRequestError, OpenAI


LMSYS_SYSTEM_TEMPLATE = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie."""


LMSYS_PROMPT_TEMPLATE = """"[User Question]\n{prompt}\n\n[The Start of Assistant A's Answer]\n{response1}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{response2}\n[The End of Assistant B's Answer]"""


HELPFULNESS_PROMPT = """The most important part of the task is choosing which of the Assistant's responses is more helpful and honest. Please consider the responses carefully, and use your overall judgment. Ask yourself "did this response represent a thoughtful, honest, and reasonable attempt to either directly help me, or to converse with me to elicit information that will make it easier for the Assistant to help later on in this conversation". Note that this is subtle - if the context is about a difficult personal situation, a helpful response might involve showing empathy and understanding. But if the context is about something practical, a helpful response may be a concise response, and not one that prolongs the interaction with irrelevant questions and comments. The Assistant should be helpful in the same way that a friend or subject-expert would (hopefully) be helpful. Note response containing "Human:" and/or  Assistant:" that tries to extend the conversation should be considered as not helpful and honest. You are an expert rater of helpful and honest Assistant responses. Given the context and the two responses choose the most helpful and honest response based on the definitions above.
Context - {prompt}
Response 1 - {response1}
Response 2 - {response2}
Preferred Response="""


class FutureJudge:
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


class OpenAIJudge(FutureJudge):
    def __init__(self, max_workers=8, model_name="gpt-4-turbo-preview"):
        super().__init__(max_workers=max_workers)
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=5)
        self.model_name = model_name

    def get_reply(self, content: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                max_tokens=3,
            )
            reply = response.choices[0].message.content
            return reply
        except BadRequestError as e:
            print("BadRequestError", e)
            print("Content: ", content)
            return random.choice(["0", "1"])


class PairRMJudge:
    def __init__(self):
        # from: https://huggingface.co/llm-blender/PairRM
        self.blender = llm_blender.Blender()
        self.blender.loadranker("llm-blender/PairRM", device=Accelerator().device)

    def judge_batch(self, prompts: List[str], completion_pairs: List[List[str]]) -> List[int]:
        results = []
        for prompt, completion_pair in zip(prompts, completion_pairs):
            ranks = self.blender.rank([prompt], [completion_pair])
            ranks = ranks - 1  # 0-indexed
            results.append(ranks[0][0])  # TODO: add order shuffling

        return results


class MockJudge:
    def judge_batch(self, prompts: List[str], completion_pairs: List[List[str]]) -> List[int]:
        results = []
        for _, _ in zip(prompts, completion_pairs):
            results.append(random.choice([0, 1]))

        return results
