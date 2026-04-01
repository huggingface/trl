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
Small-scale SDFT training with Qwen/Qwen3.5-0.8B.

Expected dataset formats:

1. Native TRL self-distillation format:
   - `prompt`
   - `privileged_context` containing only the extra teacher-only information

2. Demonstration-based format:
   - `prompt`
   - `golden_response`

Example:

```bash
python trl/experimental/sdft/sdft.py \
    --model_name_or_path Qwen/Qwen3.5-0.8B \
    --dataset_name your-org/your-dataset \
    --output_dir outputs/sdft-qwen3.5-0.8b \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --max_prompt_length 1024 \
    --max_completion_length 512 \
    --generate_from_teacher \
    --sync_ref_model \
    --ref_model_sync_steps 1 \
    --ref_model_mixup_alpha 0.01 \
    --eval_strategy steps \
    --eval_steps 50 \
    --report_to wandb
```
"""

import json
import os
import re
from dataclasses import dataclass, field
from string import Template
from typing import Any

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

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
from trl.experimental.sdft import SDFTConfig, SDFTTrainer
from trl.models import unwrap_model_for_generation


DEFAULT_DEMONSTRATION_TEMPLATE = Template("""Example response: $output_text""")


@dataclass
class SDFTScriptArguments(ScriptArguments):
    ref_model_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Reference teacher model. Optional for PEFT runs, where the base model is used as teacher."},
    )
    dataset_path: str | None = field(
        default=None,
        metadata={"help": "Optional local dataset path to load with `load_from_disk`. Overrides `dataset_name`."},
    )
    privileged_context_column: str = field(
        default="privileged_context",
        metadata={"help": "Column containing precomputed privileged context for SDFT."},
    )
    golden_response_column: str = field(
        default="golden_response",
        metadata={"help": "Column containing demonstration responses used to build privileged context."},
    )
    eval_num_prompts: int | None = field(
        default=8,
        metadata={"help": "Number of prompts to log during evaluation. Set to 0 to disable completion logging."},
    )
    demonstration_template: str = field(
        default=DEFAULT_DEMONSTRATION_TEMPLATE.template,
        metadata={"help": "Template used to build privileged context from demonstration content."},
    )
    tool_eval_num_examples: int | None = field(
        default=None,
        metadata={
            "help": "Optional number of eval examples to score for tool-use metrics. Defaults to the full eval split."
        },
    )
    tool_eval_max_new_tokens: int = field(
        default=256,
        metadata={"help": "Maximum completion length for task evaluation generation."},
    )


@dataclass
class ExampleSDFTConfig(SDFTConfig):
    scale_rewards: str = field(
        default="group",
        metadata={"help": "Reward normalization mode. Supported: `group`, `batch`, `none`."},
    )


def _extract_prompt_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
        for message in reversed(prompt):
            if message.get("role") == "user":
                content = message.get("content", "")
                if isinstance(content, list):
                    return " ".join(part.get("text", "") for part in content if part.get("type") == "text")
                return content
    return str(prompt)


def _stringify_golden_response(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, list):
        return "\n".join(_stringify_golden_response(item) for item in response)
    return str(response)


def _build_privileged_context(
    example: dict[str, Any], privileged_context_column: str, golden_response_column: str, template: Template
):
    if privileged_context_column in example and example[privileged_context_column] is not None:
        privileged_context = example[privileged_context_column]
    elif golden_response_column in example:
        privileged_context = template.safe_substitute(
            orig_content=_extract_prompt_text(example["prompt"]),
            output_text=_stringify_golden_response(example[golden_response_column]),
        )
    elif "teacher_prompt" in example:
        raise ValueError(
            "Datasets for `trl.experimental.sdft` should provide `privileged_context` or `golden_response`, not "
            "`teacher_prompt`."
        )
    else:
        raise ValueError("Dataset must contain either `privileged_context` or `golden_response` alongside `prompt`.")

    return {
        "prompt": example["prompt"],
        "privileged_context": privileged_context,
    }


def _prepare_split(dataset, script_args: SDFTScriptArguments):
    template = Template(script_args.demonstration_template)
    return dataset.map(
        lambda example: _build_privileged_context(
            example,
            privileged_context_column=script_args.privileged_context_column,
            golden_response_column=script_args.golden_response_column,
            template=template,
        ),
        remove_columns=dataset.column_names,
    )


def _can_prepare_privileged_context(dataset) -> bool:
    columns = set(dataset.column_names)
    return "prompt" in columns and ("privileged_context" in columns or "golden_response" in columns)


def _extract_action_and_input(text: str) -> tuple[str | None, str | None]:
    action_match = re.search(r"Action:\s*([^\n]+)", text)
    action_input_match = re.search(r"Action Input:\s*(.*)", text, flags=re.DOTALL)
    action = action_match.group(1).strip() if action_match else None
    action_input = action_input_match.group(1).strip() if action_input_match else None
    return action, action_input


def _parse_json_object(text: str | None) -> tuple[bool, Any]:
    if text is None:
        return False, None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return True, json.loads(text)
    except Exception:
        return False, None


def _normalize_gold_answer(example: dict[str, Any]) -> tuple[str | None, Any]:
    answers = example.get("golden_answer") or []
    if not answers:
        return None, None
    answer = answers[0]
    action = answer.get("Action")
    valid_json, action_input = _parse_json_object(answer.get("Action_Input"))
    return action, action_input if valid_json else answer.get("Action_Input")


def _apply_prompt_template(tokenizer, prompt: Any) -> str:
    return maybe_apply_chat_template({"prompt": prompt}, tokenizer)["prompt"]


def _run_tooluse_eval(
    trainer: SDFTTrainer,
    eval_dataset,
    max_new_tokens: int,
    num_examples: int | None = None,
    metric_prefix: str = "tool_eval",
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

    with (
        unwrap_model_for_generation(
            trainer.model_wrapped,
            trainer.accelerator,
            gather_deepspeed3_params=trainer.args.ds3_gather_for_generation,
        ) as unwrapped_model,
        torch.no_grad(),
    ):
        generated = unwrapped_model.generate(
            **tokenized,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=trainer.processing_class.pad_token_id,
            eos_token_id=trainer.processing_class.eos_token_id,
        )

    prompt_length = tokenized["input_ids"].shape[1]
    completions = trainer.processing_class.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)

    action_correct = 0
    json_valid = 0
    full_match = 0
    parsed_action_present = 0
    records = []

    for example, completion in zip(eval_dataset, completions, strict=True):
        pred_action, pred_action_input_text = _extract_action_and_input(completion)
        if pred_action is not None:
            parsed_action_present += 1
        pred_json_valid, pred_action_input = _parse_json_object(pred_action_input_text)
        if pred_json_valid:
            json_valid += 1

        gold_action, gold_action_input = _normalize_gold_answer(example)
        is_action_correct = pred_action == gold_action and gold_action is not None
        if is_action_correct:
            action_correct += 1
        is_full_match = is_action_correct and pred_json_valid and pred_action_input == gold_action_input
        if is_full_match:
            full_match += 1

        records.append(
            {
                "prompt": _extract_prompt_text(example["prompt"]),
                "completion": completion,
                "pred_action": pred_action,
                "pred_action_input_text": pred_action_input_text,
                "gold_action": gold_action,
                "gold_action_input": gold_action_input,
                "action_correct": is_action_correct,
                "json_valid": pred_json_valid,
                "full_match": is_full_match,
            }
        )

    total = max(len(eval_dataset), 1)
    metrics = {
        f"{metric_prefix}/action_present_rate": parsed_action_present / total,
        f"{metric_prefix}/valid_json_rate": json_valid / total,
        f"{metric_prefix}/action_accuracy": action_correct / total,
        f"{metric_prefix}/tool_call_accuracy": full_match / total,
    }

    sample_path = os.path.join(trainer.args.output_dir, f"{metric_prefix}_samples.json")
    os.makedirs(trainer.args.output_dir, exist_ok=True)
    with open(sample_path, "w") as f:
        json.dump(records[: min(20, len(records))], f, indent=2)

    return metrics


if __name__ == "__main__":
    parser = TrlParser((SDFTScriptArguments, ExampleSDFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    if model_args.model_name_or_path is None:
        raise ValueError("`model_name_or_path` is required.")
    if script_args.ref_model_name_or_path is None and not model_args.use_peft:
        script_args.ref_model_name_or_path = model_args.model_name_or_path

    if model_args.dtype in ["auto", None]:
        if training_args.bf16:
            dtype = torch.bfloat16
        elif training_args.fp16:
            dtype = torch.float16
        else:
            dtype = "auto"
    else:
        dtype = getattr(torch, model_args.dtype)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    training_args.model_init_kwargs = model_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if script_args.dataset_path is not None:
        dataset = load_from_disk(script_args.dataset_path)
    else:
        dataset = load_dataset(
            script_args.dataset_name,
            name=script_args.dataset_config,
            streaming=script_args.dataset_streaming,
        )

    if not isinstance(dataset, DatasetDict):
        raise ValueError("SDFT example expects a dataset with named splits.")

    train_dataset = _prepare_split(dataset[script_args.dataset_train_split], script_args)
    raw_eval_dataset = dataset[script_args.dataset_test_split] if script_args.dataset_test_split in dataset else None
    eval_dataset = None
    if (
        training_args.eval_strategy != "no"
        and raw_eval_dataset is not None
        and _can_prepare_privileged_context(raw_eval_dataset)
    ):
        eval_dataset = _prepare_split(raw_eval_dataset, script_args)

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    ref_model = None
    if script_args.ref_model_name_or_path is not None:
        ref_model = AutoModelForCausalLM.from_pretrained(script_args.ref_model_name_or_path, **model_kwargs)
    model.config.use_cache = False if training_args.gradient_checkpointing else True
    if ref_model is not None:
        ref_model.config.use_cache = True

    trainer = SDFTTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
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

    pretrain_metrics = None
    if raw_eval_dataset is not None and "golden_answer" in raw_eval_dataset.column_names:
        pretrain_metrics = _run_tooluse_eval(
            trainer,
            raw_eval_dataset,
            max_new_tokens=script_args.tool_eval_max_new_tokens,
            num_examples=script_args.tool_eval_num_examples,
            metric_prefix="tool_eval_before",
        )
        trainer.log(pretrain_metrics)
        trainer.log_metrics("eval", pretrain_metrics)
        trainer.save_metrics("eval", pretrain_metrics)

    trainer.train()

    trainer.save_model(training_args.output_dir)
    if eval_dataset is not None:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    if raw_eval_dataset is not None and "golden_answer" in raw_eval_dataset.column_names:
        post_metrics = _run_tooluse_eval(
            trainer,
            raw_eval_dataset,
            max_new_tokens=script_args.tool_eval_max_new_tokens,
            num_examples=script_args.tool_eval_num_examples,
            metric_prefix="tool_eval_after",
        )
        if pretrain_metrics is not None:
            for key, value in pretrain_metrics.items():
                after_key = key.replace("tool_eval_before/", "tool_eval_after/")
                if after_key in post_metrics:
                    delta_name = after_key.replace("tool_eval_after/", "tool_eval_delta/")
                    post_metrics[delta_name] = post_metrics[after_key] - value
        trainer.log(post_metrics)
        trainer.log_metrics("eval", post_metrics)
        trainer.save_metrics("eval", post_metrics)

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name or script_args.dataset_path)
