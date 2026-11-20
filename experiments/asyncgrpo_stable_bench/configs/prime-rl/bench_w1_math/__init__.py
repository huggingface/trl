"""W1 math taskset for prime-rl (verifiers v1): the shared DAPO-Math-17k parquet, the shared prompt, the shared reward.

Selected with `env.taskset.id = "bench_w1_math"`; the launch script puts `configs/prime-rl` on PYTHONPATH.
Mirrors `math_env` from prime-envs (single user turn, null harness) but with `workloads/math/{prompt,reward}`.
"""

import os
import sys

import verifiers.v1 as vf


BENCH = os.environ.get("BENCH", "/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench")
sys.path.insert(0, os.path.join(BENCH, "workloads", "math"))
from reward import PROMPT, compute_score  # noqa: E402


class W1MathData(vf.TaskData):
    answer: str


class W1MathTask(vf.Task[W1MathData]):
    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        # The shared reward reads the last \boxed{} of the whole completion. The renderer splits a completion into
        # reasoning_content (inside <think>) and content; joining them in order gives the same last \boxed{}.
        msgs = trace.assistant_messages
        if not msgs:
            return 0.0
        text = (msgs[-1].reasoning_content or "") + "\n" + (msgs[-1].content or "")
        return compute_score(text, self.data.answer)


class W1MathConfig(vf.TasksetConfig):
    data_path: str = os.path.join(BENCH, "workloads", "math", "data", "train.parquet")


class W1MathTaskset(vf.Taskset[W1MathTask, W1MathConfig]):
    def load(self) -> list[W1MathTask]:
        import pandas as pd

        df = pd.read_parquet(self.config.data_path)
        return [
            W1MathTask(W1MathData(idx=i, prompt=PROMPT.format(problem=p), answer=str(a)), self.config.task)
            for i, (p, a) in enumerate(zip(df["problem"], df["answer"], strict=False))
        ]


__all__ = ["W1MathTaskset"]
