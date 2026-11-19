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

# ruff: noqa: T201

"""
Train a tiny random-weight copy of a new model for 2 steps in each TRL trainer.

The tiny model is built from the real config: same architecture class, vocabulary, special tokens, tied embeddings,
RoPE, layer types (one layer of each), experts (4, top-2) and vision tower (2 layers); only widths and depths shrink.
No weights are downloaded. Prints one row per stage: PASS, FAIL, GAP or INFO.

    python smoke_trainers.py <model_id_or_dir> [--revision <sha>] [--output-dir <dir>] [--stages sft,dpo,...]
"""

import argparse
import json
import re
from functools import partial
from pathlib import Path

import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PretrainedConfig,
)

import trl
from trl import (
    DistillationConfig,
    DistillationTrainer,
    DPOConfig,
    DPOTrainer,
    GRPOConfig,
    GRPOTrainer,
    KTOConfig,
    KTOTrainer,
    RewardConfig,
    RewardTrainer,
    RLOOConfig,
    RLOOTrainer,
    SFTConfig,
    SFTTrainer,
)
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
from trl.trainer.utils import add_fused_lm_head, compute_flops_per_token


parser = argparse.ArgumentParser()
parser.add_argument("model", help="Hub model id or local directory")
parser.add_argument("--revision", default=None, help="Hub revision (commit sha) to pin")
parser.add_argument("--output-dir", default="tiny_models", help="where the tiny model and runs are written")
parser.add_argument("--stages", default=None, help="comma-separated subset of stages to run")
args = parser.parse_args()

# Fields set on the text config when present: widths shrink, structure (counts of each kind) is kept
TEXT_SIZES = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "moe_intermediate_size": 32,
    "shared_expert_intermediate_size": 32,
    "expert_intermediate_size": 32,
    "block_ff_dim": 128,
    "num_attention_heads": 4,
    "head_dim": 16,  # multiple of 8 for Flash Attention, at least 16 for vLLM
    "global_head_dim": 32,
    "num_global_key_value_heads": 1,
    "q_lora_rank": 32,
    "kv_lora_rank": 32,
    "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 16,
    "v_head_dim": 32,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "linear_key_head_dim": 16,
    "linear_value_head_dim": 16,
    "num_experts": 4,
    "num_local_experts": 4,
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "top_k_experts": 2,
    "n_group": 1,
    "topk_group": 1,
    "first_k_dense_replace": 1,
}

config = AutoConfig.from_pretrained(args.model, revision=args.revision)
is_vlm = "vision_config" in config.sub_configs  # same rule as TRL
# The text config as shipped, before transformers derives values from it (e.g. per-layer head dimensions)
raw, _ = PretrainedConfig.get_config_dict(args.model, revision=args.revision)
reference = raw.get("text_config", raw)
quantization = getattr(config, "quantization_config", None)
loaded = config.get_text_config().to_dict()  # with the class defaults, as TRL reads it
sizes = {k: v for k, v in TEXT_SIZES.items() if reference.get(k) is not None}
# Keep MHA models MHA (MLA needs it) and GQA models GQA
sizes["num_key_value_heads"] = 4 if reference.get("num_key_value_heads") == reference.get("num_attention_heads") else 2
if reference.get("layer_types"):
    sizes["layer_types"] = list(dict.fromkeys(reference["layer_types"]))  # one layer of each type, in order
    if reference.get("full_attention_interval"):
        sizes["full_attention_interval"] = len(sizes["layer_types"])
sizes["num_hidden_layers"] = len(sizes.get("layer_types", [0, 0]))
# Rebuild the text config from the shipped fields, so that derived values (e.g. per-layer head dimensions) follow
fields = {k: v for k, v in reference.items() if k != "quantization_config"}
text_config = type(config.get_text_config()).from_dict({**fields, **sizes})
if is_vlm:
    config.text_config = text_config
    for key in ("num_hidden_layers", "depth"):
        if getattr(config.vision_config, key, None):
            setattr(config.vision_config, key, 2)
    if getattr(config.vision_config, "out_hidden_size", None):
        config.vision_config.out_hidden_size = text_config.hidden_size
    if quantization is not None:
        del config.quantization_config  # the tiny model is built in bf16
else:
    config = text_config

tiny_dir = Path(args.output_dir) / f"tiny-{config.architectures[0]}"
torch.manual_seed(0)
model_class = getattr(transformers, config.architectures[0])  # the class TRL loads a model id with
model = model_class(config).to(torch.bfloat16)
processing_class = (AutoProcessor if is_vlm else AutoTokenizer).from_pretrained(args.model, revision=args.revision)
tokenizer = processing_class.tokenizer if is_vlm else processing_class
model.save_pretrained(tiny_dir)
processing_class.save_pretrained(tiny_dir)
try:
    GenerationConfig.from_pretrained(args.model, revision=args.revision).save_pretrained(tiny_dir)
except OSError:
    pass  # no generation_config.json: transformers derives one from config.json, as for the real model
print(f"model: {args.model}  revision: {args.revision}  vlm: {is_vlm}")
print(
    f"trl {trl.__version__} ({Path(trl.__file__).parent}), "
    f"transformers {transformers.__version__} ({Path(transformers.__file__).parent}), torch {torch.__version__}"
)
print(f"tiny model: {tiny_dir}  {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
print(f"text config: {json.dumps(sizes)}")

rows = []


def report(stage, status, detail="", owner=""):
    rows.append((stage, status, str(detail).replace("\n", " ").replace("|", "\\|")[:300], owner))


# --- Static checks on the real config ---
head_dims = {k: v for k, v in reference.items() if k.endswith("head_dim") and isinstance(v, int) and v > 256}
if head_dims:
    report(
        "head_dim <= 256 (Flash Attention 3)",
        "GAP",
        f"{head_dims}: FA kernels reject it; padding-free and AsyncGRPO tree attention need them",
        "kernels/provider",
    )
kinds = set(reference.get("layer_types") or []) - {"full_attention", "sliding_attention"}
if kinds:
    supported = kinds <= {"linear_attention", "conv"}
    report(
        "AsyncGRPO tree packing layer types", "INFO" if supported else "GAP", sorted(kinds), "" if supported else "TRL"
    )
if any(name.endswith("sinks") for name, _ in model.named_parameters()):
    report("attention sinks", "GAP", "AsyncGRPO tree attention refuses attention sinks", "TRL")
if quantization is not None:
    report("quantized reference", "INFO", f"{quantization}: run the real-weights load check (dequantized by default)")
if compute_flops_per_token(config.get_text_config(), 4096) is None:
    report("FLOPs / MFU metric", "INFO", f"model_type {loaded['model_type']} not in compute_flops_per_token")
moe = any(loaded.get(k) for k in ("num_experts", "num_local_experts", "n_routed_experts"))
aux = "output_router_logits" in loaded  # what TRL checks to enable the aux loss
if moe:
    report("MoE aux loss available", "INFO", "yes" if aux else "no output_router_logits: no balancing loss in TRL")

# --- Stages ---
out = Path(args.output_dir) / "runs" / tiny_dir.name


def common(cls, **kwargs):
    # 2 steps; tiny gradients need a high learning rate for every trained parameter to move
    return cls(
        output_dir=str(out),
        max_steps=2,
        per_device_train_batch_size=2,
        learning_rate=0.1,
        report_to="none",
        save_strategy="no",
        logging_steps=1,
        bf16=True,
        **kwargs,
    )


def zen(name, image=False):
    return load_dataset("trl-internal-testing/zen-image" if image else "trl-internal-testing/zen", name, split="train")


def train(trainer):
    # Every trainable parameter must move. On text-only data a VLM's vision and audio towers get no gradient.
    before = {n: p.clone() for n, p in trainer.model.named_parameters() if p.requires_grad}
    trainer.train()
    images = "image" in trainer.train_dataset.column_names or "images" in trainer.train_dataset.column_names
    towers = re.compile(r"vision|visual|image|multi_modal|mm_projector|audio")
    unchanged = [n for n, p in before.items() if torch.equal(p, trainer.model.get_parameter(n))]
    unchanged = [n for n in unchanged if images or not towers.search(n)]
    loss = trainer.state.log_history[-1]["train_loss"]
    if unchanged:
        return "FAIL", f"{len(unchanged)}/{len(before)} trainable params unchanged, e.g. {unchanged[:3]}"
    return "PASS", f"loss {loss:.3f}"


def str_length_reward(completions, **kwargs):
    return [float(len(c[0]["content"])) for c in completions]


def fused_lm_head():
    m = model_class.from_pretrained(tiny_dir, dtype=torch.float32).cuda()
    ids = torch.randint(0, m.config.get_text_config().vocab_size, (2, 16), device="cuda")
    ref = torch.log_softmax(m(ids).logits[:, :-1].float(), -1).gather(-1, ids[:, 1:, None])[..., 0]
    add_fused_lm_head(m)
    err = (m(ids, labels=ids, fused_lm_head=True).log_probs - ref).abs().max().item()
    return ("PASS" if err < 1e-3 else "FAIL"), f"max |dlogp| {err:.1e}"


def packed_isolation():
    # Swap test on TRL's padding-free path: the log-probs of a document must not depend on the document packed
    # before it, in attention, conv and recurrent layers alike
    m = model_class.from_pretrained(tiny_dir, dtype=torch.bfloat16, **flash).cuda().eval()
    collator = DataCollatorForLanguageModeling(pad_token_id=0, padding_free=True)
    doc = torch.randint(0, 1000, (16,)).tolist()
    logps = []
    for before in (torch.randint(0, 1000, (24,)).tolist(), torch.randint(0, 1000, (24,)).tolist()):
        batch = collator([{"input_ids": before}, {"input_ids": doc}])
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items() if k != "labels"}
        with torch.no_grad():
            logps.append(m(**batch).logits[0, 24:].float().log_softmax(-1))
    diff = (logps[0] - logps[1]).abs().amax(-1)
    # bf16 noise stays below 1e-2 and is uniform; a leak is larger and concentrated on the document's first tokens
    detail = f"max |dlogp| when the previous document changes: first 4 tokens {diff[:4].max():.1e}, "
    detail += f"last 8 {diff[-8:].max():.1e}"
    return ("PASS" if diff.max() < 2e-2 else "FAIL"), detail


def fit(trainer_cls, config_cls, dataset, image=False, config=None, **kwargs):
    args = common(config_cls, **(config or {}))
    return train(trainer_cls(model=str(tiny_dir), args=args, train_dataset=zen(dataset, image), **kwargs))


def flatten(d, prefix=""):
    flat = {}
    for k, v in d.items():
        flat.update(flatten(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
    return flat


def sft_config_kept():
    # Training-only settings (e.g. `output_router_logits`) must not leak into the saved config. TRL saves every model
    # with its training dtype and `use_cache=False`, which is not model-specific.
    trainer = SFTTrainer(
        model=str(tiny_dir), args=common(SFTConfig), train_dataset=zen("conversational_language_modeling")
    )
    trainer.train()
    trainer.save_model(str(out / "saved"))
    saved = flatten(AutoConfig.from_pretrained(out / "saved").to_dict())
    original = flatten(AutoConfig.from_pretrained(tiny_dir).to_dict())
    ignored = {"dtype", "use_cache", "_name_or_path", "transformers_version"}
    changed = [k for k in sorted(set(saved) | set(original)) if saved.get(k) != original.get(k)]
    changed = [k for k in changed if k.split(".")[-1] not in ignored]
    if all(k.endswith("_token_id") for k in changed):  # transformers aligns special ids with the tokenizer
        return ("PASS", "") if not changed else ("INFO", f"special ids rewritten from the tokenizer: {changed}")
    return "FAIL", f"saved config differs on {changed}"


def scripted_completion(user, message):
    # Ids the model would generate for `message` after the generation prompt, up to and including its stop id. Use a
    # turn that extends the generation prompt (with reasoning if needed); else take the turn from where the two renders
    # diverge, e.g. after a prefilled empty thinking block that finished turns don't render.
    render = partial(tokenizer.apply_chat_template, tools=[multiply], tokenize=True, return_dict=False)
    prompt = render(user, add_generation_prompt=True)
    for msg in (message, {**message, "reasoning_content": "Let me think."}, message):
        full = render(user + [msg])
        n = next((i for i, (x, y) in enumerate(zip(prompt, full, strict=False)) if x != y), len(prompt))
        if n == len(prompt):
            break
    rest = full[n:]
    cut = next((i for i, t in enumerate(rest) if t in trainer_stop_ids), None)
    if cut is not None:
        return rest[: cut + 1]
    # The template ends the turn with the next message's marker, which the model generates to stop
    nxt = (
        {"role": "tool", "name": "multiply", "content": "73431"}
        if msg.get("tool_calls")
        else {"role": "user", "content": "ok"}
    )
    following = render(user + [msg, nxt])
    return rest + [next(t for t in following[n:] if t in trainer_stop_ids)]


def multiply(a: int, b: int) -> int:
    """
    Multiply two integers.

    Args:
        a: First integer.
        b: Second integer.

    Returns:
        The product.
    """
    return a * b


def grpo_tools():
    # Scripted generation in the model's own tool-call format: the first turn calls the tool, the next one answers. The
    # reward function receives the episode's completion ids (tool call, tool result, answer) as GRPO builds them.
    episodes = []

    def record(completions, completion_ids, **kwargs):
        episodes.extend(completion_ids)
        return [0.0] * len(completions)

    trainer = GRPOTrainer(
        model=str(tiny_dir),
        reward_funcs=record,
        args=common(GRPOConfig, num_generations=2, max_completion_length=256),
        train_dataset=zen("conversational_prompt_only"),
        tools=[multiply],
    )
    user = [{"role": "user", "content": "What is 123 * 597?"}]
    call = {"type": "function", "function": {"name": "multiply", "arguments": {"a": 123, "b": 597}}}
    first = scripted_completion(user, {"role": "assistant", "content": "", "tool_calls": [call]})
    answer = scripted_completion(user, {"role": "assistant", "content": "It is 73431."})

    def generate(input_ids, attention_mask, **kwargs):
        prompts = [tokenizer.decode(i[m.bool()]) for i, m in zip(input_ids, attention_mask, strict=True)]
        completions = [answer if "73431" in p else first for p in prompts]
        width = max(len(c) for c in completions)
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        padded = [c + [pad] * (width - len(c)) for c in completions]
        return torch.cat([input_ids, torch.tensor(padded, device=input_ids.device)], dim=-1)

    trainer.model.generate = generate
    trainer.train()
    logs = {k: v for entry in trainer.state.log_history for k, v in entry.items() if k.startswith("tools/")}
    if not (logs.get("tools/call_frequency", 0) > 0 and logs.get("tools/failure_frequency", 1) == 0):
        return "FAIL", json.dumps(logs)
    # The stop id that ends the tool-call turn must not be repeated by the tool-result suffix
    doubled = {t for ids in episodes for t, u in zip(ids, ids[1:], strict=False) if t == u and t in trainer_stop_ids}
    if doubled:
        return "FAIL", f"stop token doubled in the episode: {tokenizer.convert_ids_to_tokens(sorted(doubled))}"
    return "PASS", json.dumps(logs)


def text_only_vlm():
    # Many users load a VLM checkpoint as a causal LM for text-only training
    m = AutoModelForCausalLM.from_pretrained(tiny_dir, dtype=torch.bfloat16)
    trainer = SFTTrainer(
        model=m,
        args=common(SFTConfig),
        train_dataset=zen("conversational_language_modeling"),
        processing_class=AutoTokenizer.from_pretrained(tiny_dir),
    )
    return train(trainer)


def lora():
    trainer = SFTTrainer(
        model=str(tiny_dir),
        args=common(SFTConfig),
        train_dataset=zen("conversational_language_modeling"),
        peft_config=LoraConfig(r=8, target_modules="all-linear"),
    )
    status, detail = train(trainer)
    names = [n for n, _ in trainer.model.named_parameters() if "lora_" in n]
    experts = [n for n in names if ".experts." in n]
    return status, f"{detail}; {len(names)} LoRA tensors, {len(experts)} on routed experts"


def aux_loss():
    trainer = SFTTrainer(
        model=str(tiny_dir),
        args=common(SFTConfig, router_aux_loss_coef=0.01),
        train_dataset=zen("conversational_language_modeling"),
    )
    trainer.train()
    values = [entry["aux_loss"] for entry in trainer.state.log_history if "aux_loss" in entry]
    return ("PASS" if values and all(v > 0 for v in values) else "FAIL"), f"aux_loss {values}"


generation_config = GenerationConfig.from_pretrained(tiny_dir)
gen_eos = generation_config.eos_token_id
trainer_stop_ids = ([gen_eos] if isinstance(gen_eos, int) else list(gen_eos or [])) + [tokenizer.eos_token_id]
flash = {"attn_implementation": "kernels-community/flash-attn2"}  # what TRL's docs use for padding-free
lm, preference, prompts = "conversational_language_modeling", "conversational_preference", "conversational_prompt_only"
online = {"num_generations": 2, "max_completion_length": 8}
stages = {
    "fused_lm_head": ("fused LM head matches model logits", "TRL", fused_lm_head),
    "packing": ("packed documents stay independent", "transformers/TRL", packed_isolation),
    "sft": ("SFT", "TRL", lambda: fit(SFTTrainer, SFTConfig, lm)),
    "sft_config": ("SFT saves the config it loaded", "TRL", sft_config_kept),
    "sft_assistant_only": (
        "SFT assistant_only_loss",
        "provider/TRL",
        lambda: fit(SFTTrainer, SFTConfig, lm, config={"assistant_only_loss": True}),
    ),
    "sft_packing": (
        "SFT packing (BFD, flash attention)",
        "transformers/kernels",
        lambda: fit(SFTTrainer, SFTConfig, lm, config={"packing": True, "model_init_kwargs": flash}),
    ),
    "sft_padding_free": (
        "SFT padding_free (flash attention)",
        "transformers/kernels",
        lambda: fit(SFTTrainer, SFTConfig, lm, config={"padding_free": True, "model_init_kwargs": flash}),
    ),
    "dpo": ("DPO", "TRL", lambda: fit(DPOTrainer, DPOConfig, preference)),
    "kto": ("KTO", "TRL", lambda: fit(KTOTrainer, KTOConfig, "conversational_unpaired_preference")),
    "reward": (
        "Reward",
        "transformers",
        lambda: fit(RewardTrainer, RewardConfig, "conversational_implicit_prompt_preference"),
    ),
    "grpo": (
        "GRPO",
        "TRL",
        lambda: fit(GRPOTrainer, GRPOConfig, prompts, config=online, reward_funcs=str_length_reward),
    ),
    "grpo_tools": ("GRPO with tools (scripted tool call)", "TRL/provider", grpo_tools),
    "rloo": (
        "RLOO",
        "TRL",
        lambda: fit(RLOOTrainer, RLOOConfig, prompts, config=online, reward_funcs=str_length_reward),
    ),
    "distillation": (
        "Distillation",
        "TRL",
        lambda: fit(
            DistillationTrainer,
            DistillationConfig,
            prompts,
            config={"max_completion_length": 8},
            teacher_model=str(tiny_dir),
        ),
    ),
    "lora": ("SFT with LoRA (all-linear)", "PEFT/TRL", lora),
}
if moe and aux:
    stages["aux_loss"] = ("SFT with router_aux_loss_coef", "TRL/transformers", aux_loss)
if is_vlm:
    stages["text_only_vlm"] = ("SFT, VLM loaded with AutoModelForCausalLM", "TRL/transformers", text_only_vlm)
    stages["sft_vlm"] = ("SFT on images", "TRL", lambda: fit(SFTTrainer, SFTConfig, lm, image=True))
    stages["dpo_vlm"] = ("DPO on images", "TRL", lambda: fit(DPOTrainer, DPOConfig, preference, image=True))
    stages["grpo_vlm"] = (
        "GRPO on images",
        "TRL",
        lambda: fit(GRPOTrainer, GRPOConfig, prompts, image=True, config=online, reward_funcs=str_length_reward),
    )

selected = args.stages.split(",") if args.stages else list(stages)
for key in selected:
    name, owner, fn = stages[key]
    print(f"--- {name}", flush=True)
    try:
        status, detail = fn()
    except Exception as e:
        status, detail = "FAIL", f"{type(e).__name__}: {e}"
    report(name, status, detail, owner if status in ("FAIL", "GAP") else "")
    torch.cuda.empty_cache()

print("\n| Stage | Status | Detail | Owner |\n|---|---|---|---|")
for row in rows:
    print("| " + " | ".join(row) + " |")
print(f"\n{ {s: sum(r[1] == s for r in rows) for s in ('PASS', 'FAIL', 'GAP', 'INFO')} }")
