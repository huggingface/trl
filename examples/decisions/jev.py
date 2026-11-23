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
#     "transformers @ git+https://github.com/huggingface/transformers.git@refs/pull/50586/head",
# ]
# ///

"""Train JEV on typed requests with Transformers' default data collator.

python examples/decisions/jev.py --model_name_or_path autotrust/JEV-9B --dataset_name your/decision-requests \
    --output_dir output/jev --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
    --gradient_checkpointing false
"""

from functools import partial

from datasets import load_dataset
from transformers import AutoModelForDecision, AutoTokenizer, default_data_collator, set_seed
from transformers.decision import Question

from trl import ModelConfig, ScriptArguments, TrlParser
from trl.experimental.decision import DataCollatorForDecision, DecisionConfig, DecisionTrainer


def tokenize(example, collator):
    inputs, question, targets = collator.prepare_example(example)
    if targets[0]:
        raise ValueError("JEV does not support nullable decisions")
    return {
        **{key: value[0].tolist() for key, value in inputs.items() if key != "marks"},
        "marks": inputs["marks"].tolist(),
        "num_options": [len(question.options)],
        "question_type_ids": [Question.questions_types.index(question.type)],
        "labels": list(targets[1:]),
        "is_choice": question.type == "choice",
    }


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
    collator = DataCollatorForDecision(tokenizer, model.decision_config)
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    prepare = partial(tokenize, collator=collator)
    with training_args.main_process_first(desc="Preparing JEV decisions"):
        train = dataset[script_args.dataset_train_split]
        train = train.map(prepare, remove_columns=train.column_names)
        evaluation = None
        if script_args.dataset_test_split is not None:
            evaluation = dataset[script_args.dataset_test_split]
            evaluation = evaluation.map(prepare, remove_columns=evaluation.column_names)
    trainer = DecisionTrainer(
        model,
        args=training_args,
        train_dataset=train,
        eval_dataset=evaluation,
        processing_class=tokenizer,
        data_collator=default_data_collator,
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
