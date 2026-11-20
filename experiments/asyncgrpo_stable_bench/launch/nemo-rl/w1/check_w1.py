"""CPU check for the NeMo-RL W1 setup (hopper-cpu, inside the container): config loads and validates, the dataset
renders exactly the SHARED_W1 prompt, and the env scores scripted completions exactly like workloads/math/reward.py."""

import sys

import pandas as pd
import ray
from omegaconf import OmegaConf

import nemo_w1  # noqa: F401  (registers w1_math)
from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.utils import setup_response_data
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from nemo_rl.utils.outdated_config_checks import check_outdated_config
from reward import PROMPT, compute_score

register_omegaconf_resolvers()
cfg = OmegaConf.to_container(load_config(sys.argv[1]), resolve=True)
check_outdated_config(cfg)
cfg = MasterConfig(**cfg)
for k in ("grpo", "loss_fn", "checkpointing"):
    print(k, getattr(cfg, k))
p = cfg.policy
print({k: p[k] for k in ("model_name", "tokenizer", "train_global_batch_size", "train_micro_batch_size", "max_total_sequence_length", "max_grad_norm", "precision")})
print("megatron", {k: p["megatron_cfg"][k] for k in ("tensor_model_parallel_size", "sequence_parallel", "activation_checkpointing", "optimizer", "scheduler")})
print("generation", {k: p["generation"][k] for k in ("max_new_tokens", "temperature", "top_p", "top_k", "colocated")}, p["generation"]["vllm_cfg"])
print("dynamic_batching", p["dynamic_batching"]["enabled"], "sequence_packing", p["sequence_packing"]["enabled"])

ray.init(num_cpus=8, include_dashboard=False)
tok = get_tokenizer(p["tokenizer"])
ds, val, task_to_env, _ = setup_response_data(tok, cfg.data, cfg.env)
print("train rows", len(ds), "val", val)
df = pd.read_parquet("/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/workloads/math/data/train.parquet")
assert len(ds) == len(df), (len(ds), len(df))
bad, maxlen = 0, 0
for i in range(len(df)):
    d = ds[i]
    ref = tok.apply_chat_template([{"role": "user", "content": PROMPT.format(problem=df.problem[i])}], tokenize=False,
                                  add_generation_prompt=True, enable_thinking=False)
    got = d["message_log"][0]["content"]
    bad += got != ref or d["extra_env_info"]["ground_truth"] != df.answer[i] or d["loss_multiplier"] != 1.0
    maxlen = max(maxlen, d["length"])
    if i == 0:
        print(repr(got[-200:]))
        assert got.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
print(f"prompt/answer mismatches {bad} of {len(df)}, max prompt tokens {maxlen}")
assert bad == 0

env = list(task_to_env.values())[0]
ans = df.answer[0]
cases = [f"so the answer is \\boxed{{{ans}}}", "\\boxed{-12345}", f"<think>x</think> \\boxed{{{ans}}} then \\boxed{{0}}",
         f"\\boxed{{{ans}", "no box", f"\\boxed{{{ans}}}" + "\nwait\n" * 3]
logs = [[{"role": "user", "content": "q"}, {"role": "assistant", "content": c}] for c in cases]
out = ray.get(env.step.remote(logs, [{"ground_truth": ans}] * len(cases)))
want = [compute_score(c, ans) for c in cases]
print("env", out.rewards.tolist(), "reference", want, "done", out.terminateds.tolist())
assert out.rewards.tolist() == want
print("CHECK_OK")
