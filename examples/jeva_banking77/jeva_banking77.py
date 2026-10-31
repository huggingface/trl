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

# Jeva: System One Models from First Principles
# Copyright 2026 Qi Li and Xinchao Wang.

# /// script
# dependencies = [
#     "trl[peft]",
#     "transformers @ git+https://github.com/huggingface/transformers.git@refs/pull/50203/head",
#     "accelerate @ git+https://github.com/huggingface/accelerate.git",
# ]
# ///

"""
Train a Qwen decision model to select banking intents, including requests with no matching option.

python examples/jeva_banking77/jeva_banking77.py --config examples/jeva_banking77/config.yaml

accelerate launch --config_file examples/accelerate_configs/multi_gpu.yaml \
    examples/jeva_banking77/jeva_banking77.py --config examples/jeva_banking77/config.yaml
"""

import json
import random
from dataclasses import dataclass, field

import numpy as np
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, JevaConfig, JevaForDecision, set_seed
from transformers.utils import is_peft_available

from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from trl.experimental.decision import DataCollatorForJeva, DecisionConfig, DecisionTrainer


if is_peft_available():
    from peft import get_peft_model


@dataclass
class JevaScriptArguments(ScriptArguments):
    dataset_name: str = field(default="PolyAI/banking77", metadata={"help": "Intent classification dataset."})
    dataset_test_split: str | None = field(
        default=None, metadata={"help": "Evaluation split. Defaults to a stratified holdout from the training split."}
    )
    num_options: int = field(default=8, metadata={"help": "Candidate intents per decision."})
    null_fraction: float = field(
        default=0.2, metadata={"help": "Fraction of decisions with the correct intent withheld."}
    )
    projection_dim: int = field(default=128, metadata={"help": "Width of the decision head."})
    max_train_samples: int | None = field(default=None, metadata={"help": "Limit training examples for a short run."})
    max_eval_samples: int | None = field(default=None, metadata={"help": "Limit evaluation examples."})

    mode: str = field(default="head", metadata={"help": "Train head, lora, or full model."})
    max_state_length: int = field(default=9000, metadata={"help": "Maximum state/question token length."})
    max_option_length: int = field(default=8192, metadata={"help": "Maximum option token length."})
    truncation: str = field(default="error", metadata={"help": "Overflow policy: error or truncate."})
    option_chunk_size: int | None = field(default=8, metadata={"help": "Options per encoder forward."})
    router_aux_loss_coef: float | None = field(default=None, metadata={"help": "MoE auxiliary-loss coefficient."})


@dataclass
class JevaModelArguments(ModelConfig):
    model_name_or_path: str = field(default="Qwen/Qwen3-0.6B", metadata={"help": "Text encoder checkpoint."})
    lora_task_type: str = field(default="FEATURE_EXTRACTION", metadata={"help": "PEFT task type for the encoder."})


def prepare_dataset(dataset, num_options, null_fraction, seed):
    intents = dataset.features["label"].names

    def format_example(example, index):
        rng = random.Random(seed + index)
        label = example["label"]
        present = rng.random() >= null_fraction
        candidates = rng.sample([i for i in range(len(intents)) if i != label], num_options - int(present))
        if present:
            candidates.append(label)
        rng.shuffle(candidates)
        request = {
            "state": example["text"],
            "questions": {
                "intent": {
                    "type": "choice",
                    "instructions": "Which banking intent best matches this request?",
                    "criteria": {intents[i].replace("_", " "): None for i in candidates},
                    "allow_null": True,
                }
            },
        }
        return {"request": json.dumps(request), "target_index": candidates.index(label) if present else None}

    return dataset.map(format_example, with_indices=True, remove_columns=dataset.column_names)


def compute_metrics(prediction):
    return {"accuracy": float(np.mean(prediction.predictions.argmax(-1) == prediction.label_ids.argmax(-1)))}


def main(script_args, training_args, model_args):
    set_seed(training_args.seed)
    if script_args.mode not in {"head", "lora", "full"}:
        raise ValueError("mode must be head, lora, or full")
    training_args.accelerator_config.dispatch_batches = False
    training_args.accelerator_config.split_batches = False
    model_kwargs = dict(training_args.model_init_kwargs or {})
    model_kwargs.setdefault("revision", model_args.model_revision)
    model_kwargs.setdefault("dtype", model_args.dtype)
    if model_args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = model_args.attn_implementation
    encoder = AutoModel.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    if encoder.config.model_type in {"qwen3_5", "qwen3_5_moe"}:
        encoder = encoder.language_model
    model = JevaForDecision(
        JevaConfig(text_config=encoder.config, projection_dim=script_args.projection_dim), model=encoder
    )
    model.base_model.requires_grad_(script_args.mode == "full")
    model.freeze_backbone = script_args.mode == "head"
    if script_args.mode == "lora":
        peft_config = get_peft_config(model_args)
        if peft_config is None:
            raise ValueError("mode='lora' requires use_peft=True")
        model.model = get_peft_model(model.base_model, peft_config)
        model.config.adapter_config = {
            key: list(value) if isinstance(value, set) else value for key, value in peft_config.to_dict().items()
        }
        model.base_model.enable_input_require_grads()
    coef = script_args.router_aux_loss_coef
    if coef is None:
        coef = model.config.text_config.to_dict().get("router_aux_loss_coef", 0.0)
    if coef and "output_router_logits" not in model.config.text_config.to_dict():
        raise ValueError("router_aux_loss_coef requires an MoE encoder that returns router logits")
    model.router_aux_loss_coef = 0.0 if model.freeze_backbone else coef
    model.config.router_aux_loss_coef = model.router_aux_loss_coef
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    train_dataset = dataset[script_args.dataset_train_split]
    if script_args.dataset_test_split is None:
        split = train_dataset.train_test_split(test_size=1024, stratify_by_column="label", seed=training_args.seed)
        train_dataset, eval_dataset = split["train"], split["test"]
    else:
        eval_dataset = dataset[script_args.dataset_test_split]
    if script_args.max_train_samples is not None:
        train_dataset = train_dataset.shuffle(seed=training_args.seed).select(range(script_args.max_train_samples))
    if script_args.max_eval_samples is not None:
        eval_dataset = eval_dataset.shuffle(seed=training_args.seed).select(range(script_args.max_eval_samples))
    train_dataset = prepare_dataset(
        train_dataset, script_args.num_options, script_args.null_fraction, training_args.seed
    )
    eval_dataset = prepare_dataset(
        eval_dataset, script_args.num_options, script_args.null_fraction, training_args.seed + 1
    )
    collator = DataCollatorForJeva(
        tokenizer,
        model.decision_config,
        script_args.max_state_length,
        script_args.max_option_length,
        script_args.truncation,
        script_args.option_chunk_size,
    )
    with training_args.main_process_first(desc="Tokenizing datasets"):
        train_dataset = train_dataset.map(
            collator.tokenize, remove_columns=train_dataset.column_names, desc="Tokenizing train dataset"
        )
        eval_dataset = eval_dataset.map(
            collator.tokenize, remove_columns=eval_dataset.column_names, desc="Tokenizing eval dataset"
        )
    trainer = DecisionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    trainer.save_metrics("train", result.metrics)
    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
    trainer.end()


if __name__ == "__main__":
    parser = TrlParser((JevaScriptArguments, DecisionConfig, JevaModelArguments))
    main(*parser.parse_args_and_config())
