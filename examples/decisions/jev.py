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
#     "transformers @ git+https://github.com/huggingface/transformers.git@refs/pull/51382/head",
# ]
# ///

"""Train JEV on typed requests with the shared decision collator.

python examples/decisions/jev.py --model_name_or_path autotrust/JEV-9B --dataset_name your/decision-requests \
    --output_dir output/jev --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
    --activation_checkpointing false
"""

from datasets import load_dataset
from transformers import AutoModelForDecision, AutoTokenizer, set_seed

from trl import ModelConfig, ScriptArguments, TrlParser
from trl.experimental.decision import DataCollatorForDecision, DecisionConfig, DecisionTrainer


def main(script_args, training_args, model_args):
    set_seed(training_args.seed)
    if training_args.per_device_train_batch_size != 1 or training_args.per_device_eval_batch_size != 1:
        raise ValueError("JEV requires per-device train and eval batch sizes of one")

    model = AutoModelForDecision.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        dtype=model_args.dtype,
        **(training_args.model_init_kwargs or {}),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    collator = DataCollatorForDecision(
        tokenizer, model.decision_config, prepare_inputs_for_decision=model.prepare_inputs_for_decision
    )
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    trainer = DecisionTrainer(
        model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if script_args.dataset_test_split is not None else None,
        processing_class=tokenizer,
        data_collator=collator,
    )

    result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    trainer.save_metrics("train", result.metrics)
    if training_args.eval_strategy != "no":
        trainer.save_metrics("eval", trainer.evaluate())
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
    trainer.end()


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, DecisionConfig, ModelConfig))
    main(*parser.parse_args_and_config())
