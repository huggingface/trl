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

"""Information aggregation (s3, arXiv 2611.08615, §4.2, Appendices F-H): the task, the paper's prompts and tools.

Each of N agents holds one private integer 1..99, shown in English words. The team must return all N numbers in
ascending order, keeping equal numbers from different agents. It succeeds if any agent's final list is correct.

A problem id fixes the numbers (16 drawn, the first N used, so one problem serves every team size); a team seed
fixes the public names (N of agent_1..agent_64) and each agent's shuffled teammate order, so the G attempts at a
problem share the numbers but not the names (the paper samples fresh names per episode).
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass


N_NUMBERS, NAME_POOL = 16, 64
LAMBDA = 0.25  # Eq. 1 token cost

_ONES = [
    "",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def number_words(n: int) -> str:
    if not 1 <= n <= 99:
        raise ValueError(f"number must be in 1..99, got {n}")
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] + (f"-{_ONES[ones]}" if ones else "")


def count_words(n: int) -> str:
    return _ONES[n] if 1 <= n < 20 else str(n)


@dataclass
class Problem:
    problem: int
    seed: int
    names: list[str]  # agent i's public name
    numbers: list[int]  # agent i's private number
    teammates: list[list[str]]  # agent i's teammates, in its own shuffled order

    @classmethod
    def make(cls, problem: int, seed: int, n: int) -> Problem:
        if not 1 <= n <= N_NUMBERS:
            raise ValueError(f"1..{N_NUMBERS} agents, got {n}")
        src = random.Random(f"aggregation-problem-{problem}")
        numbers = [src.randint(1, 99) for _ in range(N_NUMBERS)][:n]
        rng = random.Random(seed)
        names = rng.sample([f"agent_{i}" for i in range(1, NAME_POOL + 1)], n)
        teammates = []
        for i in range(n):
            others = [names[j] for j in range(n) if j != i]
            rng.shuffle(others)
            teammates.append(others)
        return cls(problem, seed, names, numbers, teammates)

    @property
    def target(self) -> list[int]:
        return sorted(self.numbers)


def team_reward(success: bool, tokens: int, n: int, allowance: int, lam: float = LAMBDA) -> float:
    """Eq. 1: R = S (1 - lam min(1, sum_i |tau_i| / (N L)))."""
    return (1.0 - lam * min(1.0, tokens / (n * allowance))) if success else 0.0


# ---- the paper's prompt texts (Appendix G, verbatim at N = 4; "[...]" elisions are unpublished and left out) ----
FIRST_USER_TURN = (
    "Solve this problem with your team: {n_cap} agents each have one private number, written in English. "
    "Find all {n} assigned numbers through communication and return their numeric values in ascending "
    "order. Include duplicates once per owning agent. You know only your own assignment; other agents do "
    "not know it unless you tell them. Final format: #### followed by a JSON array of {n} integers. "
    "Your private number: {number}."
)
TASK = (
    "You are in a team solving a private-information number sorting task. Each of {n} agents receives "
    "one English-written number. You initially know only your own number. Share your actual assignment, "
    "explicitly read partners' messages, and associate each received number with its sender so repeated "
    "messages do not create duplicates. Collect one number per agent and sort by numeric value, retaining "
    "equal values from different agents. Every agent can submit the full list. The team earns reward if "
    "at least one final answer is correct; {reward}"
)
REWARD_TOKEN_COST = (
    "there is no message-count reward. Successful rewards are reduced by at most 0.25, in proportion to "
    "the tokens all agents generated divided by the team's total generation allowance. Failures pay no "
    "token cost, and a correct answer always earns more than a failure."
)
REWARD_NO_COST = "there is no message-count or length reward."
GUIDANCE = (
    "Agents work toward the same team objective; no specialist roles are assigned by the harness. A task "
    "may give you private information that partners do not know. Work independently when useful. "
    "Communicate when you have information, checked evidence, a counterexample, a correction, or a "
    "specific question that may change a teammate's next step. Distinguish verified facts from guesses. "
    "Read replies explicitly; sending does not read messages. Teammate names are recipients of message, "
    "never function names. Use one recipient or all. Do not count repeated claims as independent "
    "verification. If evidence conflicts, compare the underlying reasoning. Submit your own checked "
    "answer when asked. Examples are hypothetical and contain no evidence about the current problem."
)
SHARED_REWARD = (
    "All participating agents share the same reward if at least one agent active at the end submits a "
    "correct answer. There is no individual-correctness bonus."
)
IDENTITY = "Your name is {name}. Your teammates: {teammates}."  # ours: the paper does not print part 4
EXAMPLES = (  # part 5: the paper's two example calls in the model's tool-call syntax ("Example..." is ours)
    "Example tool calls:\n"
    '<tool_call>\n{"name": "message", "arguments": {"to": "all", "text": "My private number is '
    'forty-two (42). Please share your own assignment."}}\n</tool_call>\n'
    '<tool_call>\n{"name": "read_messages", "arguments": {}}\n</tool_call>'
)
NOTIFICATION_RULES = (
    "While you are working, we will immediately notify you whenever another agent sends you a message, "
    "interrupting your current generation if needed. Notifications show unread-message counts by sender. "
    "If you have no unread-message notification, there is no need to check your inbox. Use read_messages "
    "when notified; do not repeatedly poll an empty inbox. You can continue reasoning or doing useful work "
    "while waiting for a notification."
)
IDLENESS_RULES = (
    "A complete tool call runs immediately and you receive its result. If you end a response without a "
    "tool call or final answer, you become idle: no tokens are spent while waiting, and a new peer message "
    "wakes you. If everyone is idle or finished, you will be asked for your final answer once. Do not "
    "repeat statements merely to keep yourself active."
)
FINAL_REQUEST = (
    "Submit the sorted list of all {n} agents' numbers now. Reply with one line: #### followed by a JSON "
    "array of {n} integers in ascending order, retaining duplicates. The messaging phase has ended."
)
NOTICE = (
    "You have {count} unread message{s} from {senders}.\nUse read_messages to read them; "
    "this notification does not mark messages as read."
)

TOOLS = [  # Table 2
    {
        "type": "function",
        "function": {
            "name": "message",
            "description": (
                "Send plans, questions, tentative ideas, dead ends, or checked results to one "
                "available teammate or all available teammates. Label uncertainty and verification "
                "status. Delivery is immediate. Sending does not read incoming messages or wait for "
                "replies; use read_messages when notified."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "A teammate's name, or all."},
                    "text": {"type": "string"},
                },
                "required": ["to", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_messages",
            "description": "Read explicitly received unread peer messages.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_notifications",
            "description": "Check unread message counts by sender without reading messages.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def opening(name: str, teammates: list[str], n: int, number: int, token_cost: bool = True) -> list[dict]:
    """The system prompt and first user turn of one agent."""
    w = count_words(n)
    system = "\n\n".join(
        [
            TASK.format(n=w, reward=REWARD_TOKEN_COST if token_cost else REWARD_NO_COST),
            GUIDANCE,
            SHARED_REWARD,
            IDENTITY.format(name=name, teammates=", ".join(teammates)),
            EXAMPLES,
            NOTIFICATION_RULES,
            IDLENESS_RULES,
        ]
    )
    user = FIRST_USER_TURN.format(n_cap=w.capitalize(), n=w, number=number_words(number))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def notice(counts: dict[str, int]) -> str:
    total, names = sum(counts.values()), list(counts)
    senders = " and ".join(names) if len(names) <= 2 else ", ".join(names[:-1]) + ", and " + names[-1]
    return json.dumps(
        {
            "unread_by_sender": counts,
            "notice": NOTICE.format(count=total, s="" if total == 1 else "s", senders=senders),
        }
    )


TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
ANSWER = re.compile(r"####\s*(\[[^\]]*\])")


def after_thinking(text: str) -> str:
    return text.rsplit("</think>", 1)[-1]


def parse_answer(text: str) -> list[int] | None:
    """The last `#### [..]` JSON array of integers in `text`, or None."""
    for m in reversed(ANSWER.findall(text)):
        try:
            v = json.loads(m)
        except json.JSONDecodeError:
            continue
        if isinstance(v, list) and all(isinstance(x, int) and not isinstance(x, bool) for x in v):
            return v
    return None
