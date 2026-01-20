import os
import random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
from tqdm import tqdm


# Reward model name -> HF path mapping (same as moodpo.py)
REWARD_MODEL_PATHS = {
    "summary": "Tristan/gpt2_reward_summarization",
    "faithful": "CogComp/bart-faithful-summary-detector",
    "helpful": "Ray2333/gpt2-large-helpful-reward_model",
    "harmless": "Ray2333/gpt2-large-harmless-reward_model",
    "humor": "mohameddhiab/humor-no-humor",
    "deberta": "OpenAssistant/reward-model-deberta-v3-large-v2",
}


@dataclass
class ScriptArguments:
    # logging / output
    save_directory: str = field(default="./eval_runs", metadata={"help": "Output directory for logs"})
    run_name: str = field(default="moodpo_eval", metadata={"help": "Run name"})

    # dataset
    csv_path: str = field(
        default="../datasets/anthropic/anthropic_test_deduped.csv",
        metadata={"help": "Path to test CSV file containing prompts"},
    )
    prompt_column: str = field(default="prompt", metadata={"help": "CSV column name for prompt"})

    # model selection (similar to moodpo.py)
    exp_type: str = field(default="assistant", metadata={"help": "assistant or summary"})
    model_path: Optional[str] = field(default=None, metadata={"help": "Policy model path (required)"})
    use_adapter: bool = field(default=False, metadata={"help": "Set True if model_path is a LoRA adapter"})
    bf16: bool = field(default=True, metadata={"help": "Use bf16 for model/reward models when on GPU"})

    # reward models (exactly 2) + weights (same order)
    reward_models: List[str] = field(
        default_factory=lambda: ["helpful", "harmless"],
        metadata={"help": "Exactly two reward model names"},
    )
    reward_weights: Optional[List[float]] = field(
        default=None,
        metadata={"help": "Weights for reward_models in the same order (2 floats). If None, uniform weights are used."},
    )

    # eval settings
    seeds: List[int] = field(default_factory=lambda: [42, 1, 2026])
    num_prompts: int = field(default=100)
    batch_size: int = field(default=1)

    # generation (wired into moodpo-style block)
    max_new_tokens: int = field(default=128)
    max_length: int = field(default=512)
    temperature: float = field(default=0.9)
    top_p: float = field(default=0.9)
    top_k: int = field(default=0)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _annotate_prompts_with_weights(prompts: list, weights: dict[str, float]) -> list:
    if not weights:
        return prompts

    weight_str = ", ".join(f"{name}: {w:.1f}" for name, w in weights.items())
    system_instruction = f"[Begin System Instruction]\nYou are an assistant. Prioritize these objectives with the given weights: {weight_str}\n[End System Instruction]"
    print(system_instruction)

    annotated_prompts = []
    for prompt in prompts:
        if isinstance(prompt, list):  # conversational format
            prompt = [msg.copy() for msg in prompt]
            if prompt and prompt[0].get("role") == "system":
                prompt[0]["content"] = system_instruction + "\n" + prompt[0]["content"]
            else:
                prompt.insert(0, {"role": "system", "content": system_instruction})
            annotated_prompts.append(prompt)
        else:
            annotated_prompts.append(system_instruction + "\n" + prompt)

    return annotated_prompts


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    if not script_args.model_path:
        raise ValueError("model_path is required")

    if len(script_args.reward_models) != 2:
        raise ValueError("Please provide exactly 2 reward models")

    if script_args.reward_weights is None:
        reward_weights = [1.0] * len(script_args.reward_models)
    else:
        if len(script_args.reward_weights) != len(script_args.reward_models):
            raise ValueError("reward_weights must match reward_models length")
        reward_weights = script_args.reward_weights

    # Output dir (moodpo.py style)
    output_dir = os.path.join(script_args.save_directory, script_args.run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    csv_path_abs = os.path.abspath(os.path.join(os.path.dirname(__file__), script_args.csv_path))
    ds = load_dataset("csv", data_files=csv_path_abs, split="train")
    if script_args.prompt_column != "prompt":
        ds = ds.rename_column(script_args.prompt_column, "prompt")
    ds = ds.map(lambda x: {"prompt": str(x["prompt"]).strip()})
    ds = ds.filter(lambda x: x["prompt"] is not None and x["prompt"] != "")
    ds = ds.filter(lambda x: len(x["prompt"]) <= 512)
    ds = ds.filter(lambda x: len(x["prompt"]) >= 8)
    prompts_all = [x["prompt"] for x in ds]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and script_args.bf16 else None

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_path, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    if script_args.use_adapter:
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(script_args.model_path, torch_dtype=dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(script_args.model_path, torch_dtype=dtype)
    model.to(device)

    # moodpo.py (100-115) style generation config block
    model.generation_config.max_new_tokens = 128 if script_args.exp_type == "assistant" else 48
    model.generation_config.temperature = script_args.temperature
    model.generation_config.top_k = script_args.top_k
    model.generation_config.top_p = script_args.top_p
    model.generation_config.do_sample = True
    model.generation_config.begin_suppress_tokens = [tokenizer.eos_token_id]
    model.generation_config.max_length = script_args.max_length
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.unk_token_id = tokenizer.unk_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.unk_token_id = tokenizer.unk_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # Reward models
        # Reward models (moodpo.py style)
    reward_models = []
    reward_tokenizers = []
    for name in script_args.reward_models:
        if name not in REWARD_MODEL_PATHS:
            raise ValueError(f"Unknown reward model '{name}'. Options: {list(REWARD_MODEL_PATHS.keys())}")
        rm = AutoModelForSequenceClassification.from_pretrained(
            REWARD_MODEL_PATHS[name], num_labels=1, torch_dtype=dtype
        )
        rm.to(device).eval()

        rtok = AutoTokenizer.from_pretrained(REWARD_MODEL_PATHS[name], padding_side="right")
        if rtok.pad_token_id is None:
            rtok.pad_token = rtok.eos_token or tokenizer.eos_token or tokenizer.unk_token

        if rm.config.pad_token_id is None:
            rm.config.pad_token_id = rtok.pad_token_id
        reward_models.append(rm)
        reward_tokenizers.append(rtok)

    reward_weights_tensor = torch.tensor(reward_weights, device=device)

    for seed in script_args.seeds:
        set_seed(seed)
        rng = random.Random(seed)

        if len(prompts_all) >= script_args.num_prompts:
            eval_prompts_raw = rng.sample(prompts_all, script_args.num_prompts)
        else:
            eval_prompts_raw = [rng.choice(prompts_all) for _ in range(script_args.num_prompts)]

        # Use the same names as training (last part of HF model path)
        model_short_names = [REWARD_MODEL_PATHS[name].split("/")[-1] for name in script_args.reward_models]
        weight_map = dict(zip(model_short_names, reward_weights))
        eval_prompts_annotated = _annotate_prompts_with_weights(eval_prompts_raw, weight_map)

        reward_sums = [0.0, 0.0]
        seen = 0

        pbar = tqdm(total=len(eval_prompts_raw), desc=f"seed {seed}", unit="prompt")

        with torch.inference_mode():
            for i in range(0, len(eval_prompts_raw), script_args.batch_size):
                batch_prompts_raw = eval_prompts_raw[i : i + script_args.batch_size]
                batch_prompts_annotated = eval_prompts_annotated[i : i + script_args.batch_size]

                enc = tokenizer(
                    batch_prompts_annotated,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=script_args.max_length,
                )
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)

                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=model.generation_config,
                )

                input_lengths = attention_mask.sum(dim=1).tolist()
                completions = []
                for j in range(generated.size(0)):
                    completion_ids = generated[j, input_lengths[j] :]
                    completions.append(tokenizer.decode(completion_ids, skip_special_tokens=True))

                # Reward eval uses ORIGINAL prompts
                reward_scores = []
                for rm, rtok in zip(reward_models, reward_tokenizers, strict=True):
                    texts = [p + c for p, c in zip(batch_prompts_raw, completions, strict=True)]
                    r_inputs = rtok(
                        texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        add_special_tokens=False,
                    )
                    r_inputs = {k: v.to(device) for k, v in r_inputs.items()}
                    scores = rm(**r_inputs).logits[:, 0]
                    reward_scores.append(scores)

                reward_scores = torch.stack(reward_scores, dim=1)  # (B, 2)
                reward_sums[0] += reward_scores[:, 0].sum().item()
                reward_sums[1] += reward_scores[:, 1].sum().item()

                batch_n = len(batch_prompts_raw)
                seen += batch_n
                mean0 = reward_sums[0] / max(1, seen)
                mean1 = reward_sums[1] / max(1, seen)

                pbar.update(batch_n)
                pbar.set_postfix(
                    {
                        f"{script_args.reward_models[0]}": f"{mean0:.2f}",
                        f"{script_args.reward_models[1]}": f"{mean1:.2f}",
                    }
                )

        pbar.close()
        n = len(eval_prompts_raw)
        print(f"\nSeed {seed} means over {n} prompts:")
        print(f"  {script_args.reward_models[0]}: {reward_sums[0] / n:.4f}")
        print(f"  {script_args.reward_models[1]}: {reward_sums[1] / n:.4f}")


if __name__ == "__main__":
    main()