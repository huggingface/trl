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
#     "trackio",
#     "kernels",
# ]
# ///

"""
Usage:

CLI:

python trl/experimental/sdpo/sdpo.py \
    --model_name_or_path Qwen/Qwen3.5-2B \
    --dataset_name HuggingFaceTB/SDPO \
    --dataset_config sciknoweval_physics \
    --output_dir outputs/sdpo-qwen35-2b-sciknoweval-physics \
    --learning_rate 1e-5 \
    --dtype bfloat16 \
    --max_completion_length 128 \
    --use_peft \
    --lora_target_modules q_proj v_proj \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_generations 4 \
    --generation_batch_size 4 \
    --distillation_alpha 1.0 \
    --full_logit_distillation false \
    --sdpo_policy_loss_mode distillation_only

YAML config:

python trl/experimental/sdpo/sdpo.py \
    --config trl/experimental/sdpo/sciknoweval_physics.yaml

This example uses the `HuggingFaceTB/SDPO` `sciknoweval_physics` subset and reports MCQ answer accuracy before and
after training. `TrlParser` will load any top-level YAML keys passed with `--config`, and command-line flags still
override the YAML values. If your dataset already contains textual environment feedback, pass the column name via
`--feedback_column`; it will be forwarded as `privileged_context` for SDPO reprompting.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Any

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer, GenerationConfig

from trl import (
    LogCompletionsCallback,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.data_utils import maybe_apply_chat_template
from trl.experimental.sdpo import SDPOConfig, SDPOTrainer


os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


DEFAULT_SYSTEM_PROMPT = (
    "Given a question and four options, please select the right answer. Respond in the following format:\n"
    "<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>\n\n"
    "For the answer, only output the letter corresponding to the correct option (A, B, C, or D), and nothing else."
)


@dataclass
class SDPOScriptArguments(ScriptArguments):
    dataset_name: str | None = field(
        default="HuggingFaceTB/SDPO",
        metadata={"help": "Dataset name. Defaults to `HuggingFaceTB/SDPO`."},
    )
    dataset_config: str | None = field(
        default="sciknoweval_physics",
        metadata={"help": "Dataset config/subset name. Defaults to `sciknoweval_physics`."},
    )
    dataset_path: str | None = field(
        default=None,
        metadata={"help": "Optional local dataset path to load with `load_from_disk`. Overrides dataset defaults."},
    )
    feedback_column: str | None = field(
        default=None,
        metadata={
            "help": "Optional dataset column containing textual environment feedback to pass as `privileged_context`."
        },
    )
    eval_num_prompts: int | None = field(
        default=8,
        metadata={"help": "Number of prompts to log during evaluation. Set to 0 to disable completion logging."},
    )
    accuracy_eval_num_examples: int | None = field(
        default=128,
        metadata={"help": "Optional number of eval examples to score for answer accuracy. Defaults to 128."},
    )
    accuracy_eval_max_new_tokens: int = field(
        default=128,
        metadata={"help": "Maximum completion length for answer-accuracy evaluation generation."},
    )
@dataclass
class ExampleSDPOConfig(SDPOConfig):
    scale_rewards: str = field(
        default="group",
        metadata={"help": "Reward normalization mode. Supported: `group`, `batch`, `none`."},
    )


def _make_conversation(example: dict[str, Any], feedback_column: str | None) -> dict[str, Any]:
    prompt = example.get("messages")
    if prompt is None:
        prompt = example.get("prompt")
        if isinstance(prompt, str):
            prompt = [
                {"role": "system", "content": example.get("system") or DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
    if prompt is None and "problem" in example:
        prompt = [
            {"role": "system", "content": example.get("system") or DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ]
    if prompt is None and "question" in example:
        prompt = [
            {"role": "system", "content": example.get("system") or DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ]

    if prompt is None:
        raise ValueError("Each example must provide one of: `messages`, `prompt`, `problem`, or `question`.")

    output = {"prompt": prompt}

    if "answer" in example:
        output["answer"] = _normalize_mcq_answer(example["answer"])
    elif "solution" in example:
        output["answer"] = _normalize_mcq_answer(example["solution"])
    else:
        raise ValueError("Each example must provide an `answer` or `solution` column for MCQ supervision.")

    if feedback_column is not None and feedback_column in example:
        output["privileged_context"] = example[feedback_column]
    elif "privileged_context" in example:
        output["privileged_context"] = example["privileged_context"]

    return output


def _apply_prompt_template(tokenizer, prompt: Any) -> str:
    return maybe_apply_chat_template({"prompt": prompt}, tokenizer)["prompt"]


def _normalize_mcq_answer(answer_text: str) -> str:
    tagged_match = re.search(r"<answer>\s*([A-D])\s*</answer>", answer_text, flags=re.IGNORECASE | re.DOTALL)
    if tagged_match is not None:
        return tagged_match.group(1).upper()

    bare_match = re.search(r"\b([A-D])\b", answer_text, flags=re.IGNORECASE)
    if bare_match is not None:
        return bare_match.group(1).upper()

    return answer_text.strip().upper()


def _extract_answer_from_tags(completion_text: str) -> str | None:
    match = re.search(r"<answer>\s*([A-D])\s*</answer>", completion_text, flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        return None
    return match.group(1).upper()


def _mcq_accuracy_reward(completions, answer, **kwargs) -> list[float]:
    rewards = []
    for completion, gold in zip(completions, answer, strict=True):
        content = completion[0]["content"] if isinstance(completion, list) else completion
        pred = _extract_answer_from_tags(content)
        rewards.append(1.0 if pred is not None and pred == _normalize_mcq_answer(gold) else 0.0)
    return rewards


def _mcq_soft_format_reward(completions, **kwargs) -> list[float]:
    pattern = r"\s*<reasoning>.*?</reasoning>\s*<answer>.*?</answer>\s*"
    rewards = []
    for completion in completions:
        content = completion[0]["content"] if isinstance(completion, list) else completion
        rewards.append(0.25 if re.fullmatch(pattern, content, flags=re.DOTALL) else 0.0)
    return rewards


def _run_accuracy_eval(
    trainer: SDPOTrainer, eval_dataset, max_new_tokens: int, num_examples: int | None, metric_prefix: str = "mcq_eval"
) -> dict[str, float]:
    if num_examples is not None:
        eval_dataset = eval_dataset.select(range(min(num_examples, len(eval_dataset))))

    prompts = eval_dataset["prompt"]
    prompt_texts = [_apply_prompt_template(trainer.processing_class, prompt) for prompt in prompts]
    tokenized = trainer.processing_class(
        text=prompt_texts,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        truncation=True,
        max_length=trainer.max_prompt_length,
        add_special_tokens=False,
    )
    tokenized = {key: value.to(trainer.accelerator.device) for key, value in tokenized.items()}
    model = trainer.accelerator.unwrap_model(trainer.model)
    with torch.no_grad():
        generated = model.generate(
            **tokenized,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=trainer.processing_class.pad_token_id,
            eos_token_id=trainer.processing_class.eos_token_id,
        )

    prompt_length = tokenized["input_ids"].shape[1]
    completions = trainer.processing_class.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)
    completion_messages = [[{"role": "assistant", "content": completion}] for completion in completions]
    rewards = _mcq_accuracy_reward(completion_messages, answer=eval_dataset["answer"])
    total = max(len(rewards), 1)
    return {
        f"{metric_prefix}/accuracy": sum(rewards) / total,
        f"{metric_prefix}/num_scored": float(len(rewards)),
    }


if __name__ == "__main__":
    parser = TrlParser((SDPOScriptArguments, ExampleSDPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        training_args.model_init_kwargs["device_map"] = get_kbit_device_map()
        training_args.model_init_kwargs["quantization_config"] = quantization_config

    if script_args.dataset_path is not None:
        dataset = load_from_disk(script_args.dataset_path)
    else:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    if not isinstance(dataset, DatasetDict):
        raise ValueError("SDPO example expects a dataset with named splits.")

    train_dataset = dataset[script_args.dataset_train_split].map(
        lambda example: _make_conversation(example, script_args.feedback_column),
        remove_columns=dataset[script_args.dataset_train_split].column_names,
    )
    eval_dataset = None
    if training_args.eval_strategy != "no":
        eval_dataset = dataset[script_args.dataset_test_split].map(
            lambda example: _make_conversation(example, script_args.feedback_column),
            remove_columns=dataset[script_args.dataset_test_split].column_names,
        )

    reward_funcs = [_mcq_soft_format_reward, _mcq_accuracy_reward]

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainer = SDPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
        processing_class=tokenizer,
    )

    if eval_dataset is not None and script_args.eval_num_prompts:
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_completion_length,
            do_sample=True,
            temperature=training_args.temperature,
        )
        trainer.add_callback(
            LogCompletionsCallback(trainer, generation_config, num_prompts=script_args.eval_num_prompts)
        )

    if eval_dataset is not None:
        pre_metrics = _run_accuracy_eval(
            trainer,
            eval_dataset,
            max_new_tokens=script_args.accuracy_eval_max_new_tokens,
            num_examples=script_args.accuracy_eval_num_examples,
        )
        trainer.log_metrics("eval", {f"before_{k}": v for k, v in pre_metrics.items()})
        trainer.save_metrics("eval", {f"before_{k}": v for k, v in pre_metrics.items()})

    trainer.train()

    trainer.save_model(training_args.output_dir)
    if eval_dataset is not None:
        post_metrics = _run_accuracy_eval(
            trainer,
            eval_dataset,
            max_new_tokens=script_args.accuracy_eval_max_new_tokens,
            num_examples=script_args.accuracy_eval_num_examples,
        )
        before_metrics = {f"before_{k}": v for k, v in pre_metrics.items()}
        after_metrics = {f"after_{k}": v for k, v in post_metrics.items()}
        delta_metrics = {
            f"delta_{k.split('/', 1)[1]}": after_metrics[f"after_{k}"] - before_metrics[f"before_{k}"]
            for k in pre_metrics
        }
        trainer.log_metrics("eval", after_metrics | delta_metrics)
        trainer.save_metrics("eval", after_metrics | delta_metrics)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name or script_args.dataset_path)
