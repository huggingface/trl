# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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

"""The environment shared by the training and evaluation scripts: a HotpotQA question and its library of articles."""

import math
import re
import string

from datasets import load_dataset


PROMPT = (
    "Answer the question below. The facts you need are in a small library of Wikipedia articles that you can only "
    "reach through your tools: search it, read the articles that matter, and follow what one article tells you to the "
    "next. When you are sure, call `submit_answer` with a short answer (a name, a date, a number, or yes/no).\n\n"
    "Question: {question}"
)


def normalize(text):
    # SQuAD answer normalization: lowercase, no punctuation, no articles, single spaces
    text = "".join(char for char in text.lower() if char not in string.punctuation)
    return " ".join(word for word in text.split() if word not in ("a", "an", "the"))


def load_questions(split):
    # Each HotpotQA question comes with ten Wikipedia paragraphs: the two that hold the answer and eight distractors
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    return dataset.map(
        lambda example: {"prompt": [{"role": "user", "content": PROMPT.format(question=example["question"])}]},
        remove_columns=["id", "type", "level", "supporting_facts"],
    )


class WikiLibrary:
    def reset(self, context, answer, **kwargs) -> None:
        self.articles = {
            title: "".join(sentences).strip()
            for title, sentences in zip(context["title"], context["sentences"], strict=True)
        }
        self.words = {
            title: set(re.findall(r"\w+", f"{title} {text}".lower())) for title, text in self.articles.items()
        }
        self.answer = answer
        self.submitted = None

    def search(self, query: str) -> str:
        """
        Search the library for articles about a topic.

        Args:
            query: What to look for: a name, a title, or a few keywords.

        Returns:
            The titles of the three best matching articles, each with its first sentence.
        """
        # Query words weighted by their rarity across the library, so "the" or "film" do not decide the ranking
        query_words = set(re.findall(r"\w+", query.lower()))
        rarity = {
            word: math.log((1 + len(self.words)) / (1 + sum(word in words for words in self.words.values())))
            for word in query_words
        }
        titles = sorted(self.words, key=lambda title: -sum(rarity[word] for word in query_words & self.words[title]))
        return "\n".join(f"{title}: {self.articles[title].split('. ')[0]}." for title in titles[:3])

    def read(self, title: str) -> str:
        """
        Read an article of the library.

        Args:
            title: The exact title of the article, as returned by `search`.

        Returns:
            The text of the article.
        """
        if title not in self.articles:
            raise ValueError(f"No article is titled {title!r}. Use `search` to find the titles.")
        return self.articles[title]

    def submit_answer(self, answer: str) -> str:
        """
        Submit the final answer to the question.

        Args:
            answer: A short answer: a name, a date, a number, or yes/no.

        Returns:
            A confirmation.
        """
        self.submitted = answer
        return "Answer submitted."

    def get_reward(self) -> float:
        return float(self.submitted is not None and normalize(self.submitted) == normalize(self.answer))
