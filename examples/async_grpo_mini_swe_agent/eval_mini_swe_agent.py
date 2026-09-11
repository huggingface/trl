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

# /// script
# dependencies = [
#     "trl",
#     "peft",
#     "datasets",
#     "openai",
#     "mini-swe-agent",
#     "huggingface_hub>=1.22",
#     "openenv @ git+https://github.com/huggingface/OpenEnv.git",
#     "swegym @ git+https://github.com/SWE-Gym/SWE-Bench-Package.git",
# ]
# ///

"""Evaluate a policy with `mini-swe-agent` on SWE-Gym, no training: the same sessions the trainer rolls out, run
against a vLLM server, graded by the same verifier. Prints the resolved rate, the per-instance pass counts and how
many instances have mixed outcomes, which is the signal GRPO trains on. Writes one JSON line per rollout (exit status,
steps, resolved, test report, transcript, submitted patch).

```sh
vllm serve Qwen/Qwen3-32B --tensor-parallel-size 2 --max-model-len 32768 \
    --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 --generation-config vllm

python examples/async_grpo_mini_swe_agent/eval_mini_swe_agent.py --model Qwen/Qwen3-32B \
    --n-instances 32 --samples-per-instance 4 --output eval.jsonl
```
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from async_grpo_mini_swe_agent import MiniSWEAgentSessionFactory, build_dataset


def run_one(factory: MiniSWEAgentSessionFactory, prompt: list[dict], sample: int) -> dict:
    t0 = time.perf_counter()
    result = {
        "instance_id": factory.instances[prompt[-1]["content"]]["instance_id"],
        "sample": sample,
        "exit_status": None,
        "steps": 0,
        "resolved": False,
        "metrics": {},
        "submission": "",
        "messages": [],
    }
    session = None
    try:
        session = factory.create(prompt)
        session.wait_for_completion()
        trace = session.fetch_proxy_trace()
        verify = session.verify([])
        last = trace[-1] if trace else {}
        choices = (last.get("response") or {}).get("choices") or []
        exit_message = session.agent.messages[-1] if session.agent.messages else {}
        result.update(
            exit_status=session.exit_status,
            steps=session.agent.n_calls,
            resolved=verify.env_reward == 1.0,
            metrics=verify.metrics,
            submission=exit_message.get("extra", {}).get("submission", ""),
            messages=list((last.get("request") or {}).get("messages") or [])
            + ([choices[0]["message"]] if choices else []),
        )
    except Exception as e:  # a sandbox that failed to start or died mid-rollout is a failed rollout, not a failed eval
        result.update(exit_status=type(e).__name__, error=str(e)[:500])
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass
    result["wall_s"] = time.perf_counter() - t0
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-32B")
    p.add_argument("--vllm-url", default="http://localhost:8000")
    p.add_argument("--output", default="eval_mini_swe_agent.jsonl")
    p.add_argument("--n-instances", type=int, default=32)
    p.add_argument("--instances-file", default=None)  # JSON list of instance ids to evaluate instead of the first n
    p.add_argument("--instance-offset", type=int, default=0)  # skip the first N instances of the shuffled order
    p.add_argument("--samples-per-instance", type=int, default=4)
    p.add_argument("--max-inflight", type=int, default=32)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-turn-tokens", type=int, default=4096)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sandbox-flavor", default="cpu-basic")
    p.add_argument("--step-limit", type=int, default=250)
    p.add_argument("--step-timeout", type=int, default=300)
    p.add_argument("--agent-timeout", type=int, default=3600)
    p.add_argument("--eval-timeout", type=int, default=900)
    args = p.parse_args()

    instance_ids = json.load(open(args.instances_file)) if args.instances_file else None
    dataset, instances = build_dataset(args.instance_offset + args.n_instances, args.seed, instance_ids)
    dataset = dataset.select(range(args.instance_offset, len(dataset)))
    factory = MiniSWEAgentSessionFactory(
        instances=instances,
        vllm_url=args.vllm_url,
        model=args.model,
        flavor=args.sandbox_flavor,
        chat_template_kwargs={} if args.enable_thinking else {"enable_thinking": False},
        temperature=args.temperature,
        max_turn_tokens=args.max_turn_tokens,
        step_limit=args.step_limit,
        step_timeout=args.step_timeout,
        agent_timeout=args.agent_timeout,
        eval_timeout=args.eval_timeout,
    )

    jobs = [(row["prompt"], sample) for row in dataset for sample in range(args.samples_per_instance)]
    results = []
    with ThreadPoolExecutor(max_workers=args.max_inflight) as pool, open(args.output, "w") as out:
        for future in as_completed([pool.submit(run_one, factory, *job) for job in jobs]):
            result = future.result()
            results.append(result)
            out.write(json.dumps(result, default=str) + "\n")
            out.flush()

    by_instance = Counter(r["instance_id"] for r in results if r["resolved"])
    n = args.samples_per_instance
    print(f"resolved {sum(by_instance.values())}/{len(results)} rollouts, {len(by_instance)}/{len(dataset)} instances")
    print("exit statuses:", dict(Counter(r["exit_status"] for r in results)))
    print(f"mean steps: {sum(r['steps'] for r in results) / len(results):.1f}")
    for instance_id in sorted({r["instance_id"] for r in results}):
        print(f"  {by_instance[instance_id]}/{n}  {instance_id}")
    print(f"mixed-outcome instances (GRPO signal): {sum(1 for k in by_instance.values() if 0 < k < n)}/{len(dataset)}")


if __name__ == "__main__":
    main()
