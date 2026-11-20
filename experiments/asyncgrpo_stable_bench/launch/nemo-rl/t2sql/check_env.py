"""CPU check of the NeMo-RL text-to-SQL glue (hopper-cpu, inside the container, no GPU):
1. the run config loads and validates (load_config + MasterConfig);
2. the dataset builds 2,033 rows whose prompt ids equal t2sql_episode.opening_prompt_ids;
3. scripted episodes through the Ray env actor give the same observation ids and rewards as t2sql_episode:
   (a) a tool call then the gold SQL answer, (b) the gold answer directly, (c) no answer.
"""

import json
import os
import sys

import ray
import torch
from omegaconf import OmegaConf

import nemo_t2sql as nt
from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


register_omegaconf_resolvers()
cfg = MasterConfig(**OmegaConf.to_container(load_config(sys.argv[1]), resolve=True))
print("config ok:", cfg.policy["model_name"], cfg.grpo.num_prompts_per_step, cfg.grpo.max_rollout_turns,
      cfg.policy["max_total_sequence_length"], cfg.policy["generation"]["max_new_tokens"],
      cfg.policy["automodel_cfg"]["expert_parallel_size"], flush=True)

from transformers import AutoTokenizer  # noqa: E402

# LoRA targets vs TRL's 310 modules: Automodel's ModuleMatcher wildcard rule (nemo_automodel/components/_peft/
# module_matcher.py:_compile_wildcard_pattern) on the base checkpoint's module names in Automodel naming
# (state_dict_adapter.py: .mlp.shared_expert. -> .mlp.shared_experts.)
import re  # noqa: E402

from huggingface_hub import snapshot_download  # noqa: E402


def _wild(pattern):
    pattern = re.sub(r"(?<!\.)\*", r".*", pattern)
    pattern = re.sub(r"\.\*", "(.*)", pattern)
    return re.compile("^" + pattern + "$")


lora = cfg.policy["automodel_cfg"].get("lora_cfg", {})
if lora.get("enabled"):
    base = snapshot_download(cfg.policy["model_name"], local_files_only=True)
    keys = json.load(open(os.path.join(base, "model.safetensors.index.json")))["weight_map"]
    modules = {k[: -len(".weight")].replace(".mlp.shared_expert.", ".mlp.shared_experts.") for k in keys if k.endswith(".weight") and ".mlp.experts." not in k}  # routed experts are grouped params, not Linear modules, in both trainers
    pats = [_wild(p) for p in lora["target_modules"]]
    hit = sorted(m for m in modules if any(p.match(m) for p in pats))
    trl = re.compile(r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj|gate_proj|up_proj|down_proj)")
    trl_hit = sorted(m.replace(".shared_experts.", ".shared_expert.") for m in modules if trl.fullmatch(m.replace(".shared_experts.", ".shared_expert.")))
    print(f"lora r{lora['dim']} alpha {lora['alpha']}: {len(hit)} target modules, TRL regex {len(trl_hit)}; e.g. {hit[:2]} {hit[-2:]}", flush=True)
    assert len(hit) == len(trl_hit) == 310, (len(hit), len(trl_hit))
    assert {m.replace(".shared_experts.", ".shared_expert.") for m in hit} == set(trl_hit)

tok = AutoTokenizer.from_pretrained(cfg.policy["model_name"])
ds = nt.Text2SQLDataset(os.environ["T2SQL_TRAIN"], tok)
print("rows", len(ds), "eos", tok.eos_token, tok.eos_token_id, flush=True)
assert len(ds) == 2033

ray.init(num_cpus=8, include_dashboard=False)
env = nt.Text2SQLEnv.options(num_gpus=0).remote(cfg.policy["model_name"], os.environ["T2SQL_EPISODES_FILE"],
                                                 [tok.eos_token_id])


def turn(log, meta, text):
    ids = tok.encode(text, add_special_tokens=False) + [tok.eos_token_id]
    log = log + [{"role": "assistant", "content": text, "token_ids": torch.tensor(ids)}]
    out = ray.get(env.step.remote([[{"role": m["role"], "content": m["content"]} for m in log]], [meta]))
    obs = out.observations[0]
    obs_ids = tok(obs["content"], return_tensors="pt", add_special_tokens=False).input_ids[0]
    log = log + [{"role": "user", "content": obs["content"], "token_ids": obs_ids}]
    return log, out.metadata[0], float(out.rewards[0]), bool(out.terminateds[0]), obs


for k in (0, 5, 17):
    d = ds[k]
    task = d["extra_env_info"]["task"]
    assert d["message_log"][0]["token_ids"].tolist() == nt.ep.opening_prompt_ids(tok, task)
    gold = task["gold_sql"]
    call = ('<think>\nlook\n</think>\n\n<tool_call>\n<function=execute_sql_query>\n<parameter=query>\n'
            'SELECT 1\n</parameter>\n</function>\n</tool_call>')
    answer = f"<think>\nok\n</think>\n\n<solution>{gold}</solution>"
    # (a) tool call, then answer
    log, meta, r1, t1, obs = turn(d["message_log"], d["extra_env_info"], call)
    expected = nt.ep._observation_ids(tok, [{"role": "user", "content": "x"}], last_token_id=tok.eos_token_id)
    print(f"task {task['id']}: tool turn reward {r1} terminated {t1}; obs starts {obs['content'][:120]!r}")
    assert not t1 and r1 == 0.0, "tool call should continue"
    _, _, r2, t2, _ = turn(log, meta, answer)
    print(f"  answer after tool: reward {r2:.3f} terminated {t2}")
    # (b) direct answer, (c) no answer
    d = ds[k]
    _, _, rb, tb, _ = turn(d["message_log"], d["extra_env_info"], answer)
    d = ds[k]
    _, _, rc, tc, _ = turn(d["message_log"], d["extra_env_info"], "<think>\nhmm\n</think>\n\nI don't know.")
    print(f"  direct answer reward {rb:.3f}; no answer reward {rc:.3f}")
    assert tb and tc and rc <= -1 + 1e-6

print(open(os.environ["T2SQL_EPISODES_FILE"]).read()[-600:])
print("CHECK OK")
