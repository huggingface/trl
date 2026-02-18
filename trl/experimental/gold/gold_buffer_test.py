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
# ]
# ///

"""
Buffered GOLD trainer smoke-test script.

Example (CLI args):
python trl/experimental/gold/gold_buffer_test.py \
    --model_name_or_path HuggingFaceH4/KD-Thinky \
    --teacher_model_name_or_path Qwen/Qwen3-8B \
    --dataset_name HuggingFaceH4/DeepMath-103K \
    --dataset_config trl_all \
    --output_dir data/gold-buffer-test \
    --max_steps 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --steps_per_generation 4 \
    --num_generations 4 \
    --lmbda 1.0 \
    --bf16

Example (YAML config inspired by internal recipes):
python trl/experimental/gold/gold_buffer_test.py --config path/to/config.yaml
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from datasets import Dataset, DatasetDict, IterableDataset, load_dataset
from transformers import AutoTokenizer, TrainerCallback

from trl import (
    DatasetMixtureConfig,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_dataset,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.gold import GOLDConfig, GOLDTrainer


logger = logging.getLogger(__name__)


@dataclass
class GoldBufferTestArguments:
    dataset_mixture: dict[str, Any] | None = field(
        default=None,
        metadata={
            "help": (
                "Dataset mixture config. Supports both public format (`datasets`) and internal-like format "
                "(`dataset_mixture.datasets` with `id`/`config`)."
            )
        },
    )
    max_train_samples: int | None = field(
        default=64,
        metadata={"help": "Optional cap on train samples for quick smoke tests."},
    )
    max_eval_samples: int | None = field(
        default=32,
        metadata={"help": "Optional cap on eval samples for quick smoke tests."},
    )
    require_buffer_usage: bool = field(
        default=True,
        metadata={"help": "Fail if buffered generation path is not observed when steps_per_generation > 1."},
    )


class BufferSanityCallback(TrainerCallback):
    def __init__(self, trainer: GOLDTrainer, require_buffer_usage: bool = True):
        self.trainer = trainer
        self.require_buffer_usage = require_buffer_usage
        self.buffer_seen = False

    def on_step_end(self, args, state, control, **kwargs):
        steps_per_generation = max(1, int(self.trainer.args.steps_per_generation))
        if steps_per_generation <= 1:
            return control
        buffered_inputs = getattr(self.trainer, "_buffered_inputs", None)
        buffered_flags = getattr(self.trainer, "_buffered_on_policy", None)
        if (
            isinstance(buffered_inputs, list)
            and isinstance(buffered_flags, list)
            and len(buffered_inputs) == steps_per_generation
            and len(buffered_flags) == steps_per_generation
        ):
            self.buffer_seen = True
        return control

    def on_train_end(self, args, state, control, **kwargs):
        steps_per_generation = max(1, int(self.trainer.args.steps_per_generation))
        if self.require_buffer_usage and steps_per_generation > 1 and not self.buffer_seen:
            raise RuntimeError(
                "Buffer sanity check failed: trainer did not expose buffered rollout state while "
                "steps_per_generation > 1."
            )
        return control


def _normalize_internal_like_mixture(raw: dict[str, Any]) -> DatasetMixtureConfig:
    datasets_raw = raw.get("datasets", [])
    normalized_datasets = []
    for entry in datasets_raw:
        path = entry.get("path", entry.get("id"))
        name = entry.get("name", entry.get("config"))
        if path is None:
            raise ValueError(f"Each dataset entry must provide `path` or `id`. Got: {entry}")
        if "weight" in entry:
            logger.warning("Ignoring dataset `weight`=%s for %s in smoke-test script.", entry["weight"], path)
        normalized_datasets.append(
            {
                "path": path,
                "name": name,
                "data_dir": entry.get("data_dir"),
                "data_files": entry.get("data_files"),
                "split": entry.get("split", "train"),
                "columns": entry.get("columns"),
            }
        )

    return DatasetMixtureConfig(
        datasets=normalized_datasets,
        streaming=raw.get("streaming", False),
        test_split_size=raw.get("test_split_size"),
    )


def _resolve_dataset(
    script_args: ScriptArguments,
    test_args: GoldBufferTestArguments,
) -> DatasetDict:
    if test_args.dataset_mixture is not None:
        mixture = _normalize_internal_like_mixture(test_args.dataset_mixture)
        return get_dataset(mixture)

    if script_args.dataset_name is None:
        raise ValueError("Either `dataset_name` or `dataset_mixture` must be provided.")
    return load_dataset(
        script_args.dataset_name,
        name=script_args.dataset_config,
        streaming=script_args.dataset_streaming,
    )


def _cap_dataset_size(dataset: Dataset | IterableDataset, cap: int | None):
    if cap is None:
        return dataset
    if isinstance(dataset, IterableDataset):
        return dataset.take(cap)
    cap = min(cap, len(dataset))
    return dataset.select(range(cap))


def _pick_split(dataset_dict: DatasetDict, split_name: str, fallbacks: tuple[str, ...]):
    if split_name in dataset_dict:
        return dataset_dict[split_name]
    for candidate in fallbacks:
        if candidate in dataset_dict:
            logger.warning("Split `%s` not found. Falling back to `%s`.", split_name, candidate)
            return dataset_dict[candidate]
    raise ValueError(f"Split `{split_name}` not found. Available splits: {list(dataset_dict.keys())}")


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GOLDConfig, ModelConfig, GoldBufferTestArguments))
    script_args, training_args, model_args, test_args, _ = parser.parse_args_and_config(return_remaining_strings=True)

    if training_args.student_model_revision in (None, "main") and model_args.model_revision is not None:
        training_args.student_model_revision = model_args.model_revision

    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(training_args.model_init_kwargs or {})
    model_kwargs.setdefault("revision", training_args.student_model_revision)
    model_kwargs.setdefault("trust_remote_code", model_args.trust_remote_code)
    model_kwargs.setdefault("attn_implementation", model_args.attn_implementation)
    model_kwargs.setdefault("torch_dtype", model_args.dtype)
    model_kwargs.setdefault("use_cache", False if training_args.gradient_checkpointing else True)
    if quantization_config is not None:
        model_kwargs.setdefault("device_map", get_kbit_device_map())
        model_kwargs.setdefault("quantization_config", quantization_config)
    training_args.model_init_kwargs = model_kwargs

    if training_args.teacher_model_name_or_path is None:
        training_args.teacher_model_name_or_path = model_args.model_name_or_path
    if training_args.use_uld_loss and training_args.teacher_tokenizer_name_or_path is None:
        training_args.teacher_tokenizer_name_or_path = training_args.teacher_model_name_or_path

    teacher_model_kwargs = dict(training_args.teacher_model_init_kwargs or {})
    teacher_model_kwargs.setdefault("revision", model_args.model_revision)
    teacher_model_kwargs.setdefault("trust_remote_code", model_args.trust_remote_code)
    teacher_model_kwargs.setdefault("attn_implementation", model_args.attn_implementation)
    teacher_model_kwargs.setdefault("torch_dtype", model_args.dtype)
    teacher_model_kwargs.setdefault("use_cache", True)
    if quantization_config is not None:
        teacher_model_kwargs.setdefault("device_map", get_kbit_device_map())
        teacher_model_kwargs.setdefault("quantization_config", quantization_config)
    training_args.teacher_model_init_kwargs = teacher_model_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_dict = _resolve_dataset(script_args, test_args)
    train_dataset = _pick_split(dataset_dict, script_args.dataset_train_split, ("train",))
    train_dataset = _cap_dataset_size(train_dataset, test_args.max_train_samples)

    eval_dataset = None
    if training_args.eval_strategy != "no":
        eval_dataset = _pick_split(dataset_dict, script_args.dataset_test_split, ("validation", "dev", "test"))
        eval_dataset = _cap_dataset_size(eval_dataset, test_args.max_eval_samples)

    trainer = GOLDTrainer(
        model=model_args.model_name_or_path,
        teacher_model=training_args.teacher_model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    buffer_sanity = BufferSanityCallback(trainer, require_buffer_usage=test_args.require_buffer_usage)
    trainer.add_callback(buffer_sanity)

    logger.info(
        "Starting GOLD buffer test: steps_per_generation=%s, num_generations=%s, lmbda=%s, use_vllm=%s",
        training_args.steps_per_generation,
        training_args.num_generations,
        training_args.lmbda,
        training_args.use_vllm,
    )

    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    metrics = dict(train_result.metrics)
    metrics["buffer_seen"] = int(buffer_sanity.buffer_seen)
    if not isinstance(train_dataset, IterableDataset):
        metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
