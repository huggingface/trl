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
#     "math-verify",
#     "latex2sympy2_extended",
#     "trackio",
#     "kernels",
# ]
# ///

"""
Usage:

```bash
python trl/experimental/sdpo/sdpo.py \
    --model_name_or_path Qwen/Qwen2.5-Math-1.5B-Instruct \
    --dataset_name openai/gsm8k \
    --dataset_config main \
    --output_dir outputs/sdpo-qwen35-2b-gsm8k \
    --learning_rate 5e-5 \
    --dtype bfloat16 \
    --bf16 true \
    --max_completion_length 128 \
    --use_peft \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --num_generations 8 \
    --generation_batch_size 32 \
    --distillation_alpha 1.0 \
    --full_logit_distillation false \
    --sdpo_policy_loss_mode hybrid \
    --report_to none \
    --eval_strategy steps \
    --eval_steps 1000 \
    --save_strategy no \
    --eval_num_prompts 0 \
    --accuracy_eval_num_examples 64 \
    --max_train_examples 256 \
    --max_eval_examples 128
```

This example uses verifiable math rewards and reports answer accuracy before and after training. If your dataset
already contains textual environment feedback, pass the column name via `--feedback_column`; it will be forwarded as
`privileged_context` for SDPO reprompting.
"""

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


SYSTEM_PROMPT = (
    "A conversation between user and assistant. The user asks a question, and the assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "must be enclosed within <think></think> tags, and the final answer must be on its own line in the format "
    "`#### <answer>`."
)


@dataclass
class SDPOScriptArguments(ScriptArguments):
    dataset_path: str | None = field(
        default=None,
        metadata={"help": "Optional local dataset path to load with `load_from_disk`. Overrides `dataset_name`."},
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
    feedback_from_solution: str | None = field(
        default=None,
        metadata={
            "help": "Optional synthesized feedback source when the dataset has no feedback column. Supported: "
            "`final_answer`, `full_solution`."
        },
    )
    max_train_examples: int | None = field(
        default=None,
        metadata={"help": "Optional cap on the number of training examples loaded from the selected train split."},
    )
    max_eval_examples: int | None = field(
        default=None,
        metadata={"help": "Optional cap on the number of evaluation examples loaded from the selected eval split."},
    )
    dataset_shuffle_seed: int = field(
        default=42,
        metadata={"help": "Random seed used before applying `max_train_examples` or `max_eval_examples`."},
    )


@dataclass
class ExampleSDPOConfig(SDPOConfig):
    scale_rewards: str = field(
        default="group",
        metadata={"help": "Reward normalization mode. Supported: `group`, `batch`, `none`."},
    )


def _make_solution_feedback(final_answer: str, worked_solution: str, feedback_from_solution: str | None) -> str | None:
    if feedback_from_solution is None:
        return None
    if feedback_from_solution == "final_answer":
        return (
            "Your previous answer was incorrect. The correct final answer is:\n\n"
            f"#### {final_answer}\n\n"
            "Revise your reasoning and end with the same final answer format."
        )
    if feedback_from_solution == "full_solution":
        return (
            "Your previous answer was incorrect. Here is a correct worked solution:\n\n"
            f"{worked_solution}\n\n"
            "Use it to solve the original question correctly."
        )
    raise ValueError("feedback_from_solution must be one of: `final_answer`, `full_solution`.")


def _make_conversation(
    example: dict[str, Any], feedback_column: str | None, feedback_from_solution: str | None
) -> dict[str, Any]:
    prompt = example.get("prompt")
    if prompt is None and "problem" in example:
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ]
    if prompt is None and "question" in example:
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ]

    if prompt is None:
        raise ValueError("Each example must provide one of: `prompt`, `problem`, or `question`.")

    output = {"prompt": prompt}

    solution = None
    if "solution" in example:
        solution = example["solution"]
    elif "answer" in example:
        solution = _normalize_gsm8k_answer(example["answer"])

    if solution is not None:
        output["solution"] = solution

    if feedback_column is not None and feedback_column in example:
        output["privileged_context"] = example[feedback_column]
    elif "privileged_context" in example:
        output["privileged_context"] = example["privileged_context"]
    elif solution is not None:
        worked_solution = example.get("solution")
        if worked_solution is None and "answer" in example:
            worked_solution = example["answer"].strip()
        if worked_solution is None:
            worked_solution = f"#### {solution}"
        synthesized_feedback = _make_solution_feedback(solution, worked_solution, feedback_from_solution)
        if synthesized_feedback is not None:
            output["privileged_context"] = synthesized_feedback

    return output


def _normalize_gsm8k_answer(answer_text: str) -> str:
    if "####" not in answer_text:
        return answer_text.strip()
    return answer_text.split("####", 1)[1].strip().replace(",", "")


def _extract_predicted_answer(completion_text: str) -> str | None:
    match = re.search(r"####\s*([^\n]+)", completion_text)
    if match:
        return match.group(1).strip().replace(",", "")

    matches = re.findall(r"(-?\$?[0-9][0-9,]*(?:\.[0-9]+)?)", completion_text)
    if not matches:
        return None
    return matches[-1].replace("$", "").replace(",", "").strip()


def _gsm8k_accuracy_reward(completions, solution, **kwargs) -> list[float]:
    rewards = []
    for completion, gold in zip(completions, solution, strict=True):
        content = completion[0]["content"] if isinstance(completion, list) else completion
        pred = _extract_predicted_answer(content)
        rewards.append(1.0 if pred is not None and pred == gold else 0.0)
    return rewards


def _gsm8k_soft_format_reward(completions, **kwargs) -> list[float]:
    pattern = r"<think>.*?</think>\s*####\s*[^\n]+"
    rewards = []
    for completion in completions:
        content = completion[0]["content"] if isinstance(completion, list) else completion
        rewards.append(0.25 if re.match(pattern, content, flags=re.DOTALL) else 0.0)
    return rewards


def _run_accuracy_eval(
    trainer: SDPOTrainer, eval_dataset, max_new_tokens: int, num_examples: int | None, metric_prefix: str = "math_eval"
) -> dict[str, float]:
    if num_examples is not None:
        eval_dataset = eval_dataset.select(range(min(num_examples, len(eval_dataset))))

    prompts = eval_dataset["prompt"]
    prompt_texts = [
        maybe_apply_chat_template({"prompt": prompt}, trainer.processing_class)["prompt"] for prompt in prompts
    ]
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
    was_training = model.training
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            **tokenized,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=trainer.processing_class.pad_token_id,
            eos_token_id=trainer.processing_class.eos_token_id,
        )
    if was_training:
        model.train()

    prompt_length = tokenized["input_ids"].shape[1]
    completions = trainer.processing_class.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)
    completion_messages = [[{"role": "assistant", "content": completion}] for completion in completions]
    rewards = _gsm8k_accuracy_reward(completion_messages, solution=eval_dataset["solution"])
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

    train_split = dataset[script_args.dataset_train_split]
    if script_args.max_train_examples is not None:
        train_split = train_split.shuffle(seed=script_args.dataset_shuffle_seed).select(
            range(min(script_args.max_train_examples, len(train_split)))
        )

    train_dataset = train_split.map(
        lambda example: _make_conversation(example, script_args.feedback_column, script_args.feedback_from_solution),
        remove_columns=train_split.column_names,
    )
    eval_dataset = None
    if training_args.eval_strategy != "no":
        eval_split = dataset[script_args.dataset_test_split]
        if script_args.max_eval_examples is not None:
            eval_split = eval_split.shuffle(seed=script_args.dataset_shuffle_seed).select(
                range(min(script_args.max_eval_examples, len(eval_split)))
            )

        eval_dataset = eval_split.map(
            lambda example: _make_conversation(
                example, script_args.feedback_column, script_args.feedback_from_solution
            ),
            remove_columns=eval_split.column_names,
        )

    reward_funcs = [_gsm8k_soft_format_reward, _gsm8k_accuracy_reward]

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
        after_metrics = {f"after_{k}": v for k, v in post_metrics.items()}
        delta_metrics = {
            f"delta_{k.split('/', 1)[1]}": after_metrics[f"after_{k}"] - pre_metrics[k] for k in pre_metrics
        }
        trainer.log_metrics("eval", after_metrics | delta_metrics)
        trainer.save_metrics("eval", after_metrics | delta_metrics)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name or script_args.dataset_path)
