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
#     "trl[peft]>=1.16.0",  # the sequence boundaries of the padding-free collator
#     "transformers[serving]>=5.20.0",  # the prompt of the `/v1/systemone` endpoint
#     "datasets>=5.0.0",  # `Json` feature
#     "trackio",
# ]
# ///

# docstyle-ignore
"""
Trains Qwen/Qwen3.5-2B as a decision ("System One") model: it answers typed questions about a state with a probability
for each option, from the next-token logits of the option labels (`Yes`/`No`, `A`, `B`, ...) at the end of the prompt.
The prompt is the one of the `/v1/systemone` endpoint of `transformers serve`, so the merged model is served as is.
LoRA on 40k questions of tasksource-jev-typed-decisions, with their soft targets. On one RTX 4070 Laptop GPU (8 GB):
961 steps of 32 questions, 4.9 s/step, 6 GB peak; one epoch takes 78 minutes.

    python examples/sft_decision_model/1_sft.py
"""

import torch
from decisions import load_train
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cli.serving.systemone import SystemOneRequest, parse_question, render_decision_prompt

from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
from trl.trainer.utils import pad


MODEL = "Qwen/Qwen3.5-2B"
OUTPUT_DIR = "Qwen3.5-2B-Decision"
MAX_LENGTH = 512


def tokenize(example, tokenizer):
    request = SystemOneRequest.model_validate(
        {"model": MODEL, "state": example["state"], "questions": {"q": example["question"]}}
    )
    question = parse_question("q", request.questions["q"])
    messages = [{"role": "user", "content": render_decision_prompt(request.state, question)}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    return {
        "input_ids": tokenizer(prompt, add_special_tokens=False)["input_ids"],
        # Each label is a single token, which the endpoint checks
        "label_ids": [ids[0] for ids in tokenizer(question.labels, add_special_tokens=False)["input_ids"]],
        "target": example["target"],
    }


class DecisionCollator(DataCollatorForLanguageModeling):
    def torch_call(self, examples):
        batch = super().torch_call(examples)
        # Questions have different numbers of options: the missing ones are -1, with a target of 0
        batch["label_ids"] = pad([torch.tensor(example["label_ids"]) for example in examples], padding_value=-1)
        batch["target"] = pad([torch.tensor(example["target"]) for example in examples], padding_value=0.0)
        return batch


class DecisionTrainer(SFTTrainer):
    # The loss is a mean over the questions of the batch, which `Trainer` must divide by the accumulation steps
    loss_is_scaled_for_ga = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        mode = "train" if model.training else "eval"
        label_ids, target = inputs.pop("label_ids"), inputs.pop("target")
        inputs.pop("labels")
        # Padding-free: the prompts are one sequence, and only the logits at the last token of each are computed
        last_tokens = inputs["cu_seq_lens_q"][1:] - 1
        logits = model(**inputs, use_cache=False, logits_to_keep=last_tokens).logits[0].float()

        # Cross-entropy with the target probabilities over the options of each question, like the softmax the endpoint
        # applies to their logits
        is_option = label_ids >= 0
        option_logits = logits.gather(-1, label_ids.clamp(min=0)).masked_fill(~is_option, float("-inf"))
        logps = option_logits.log_softmax(-1).masked_fill(~is_option, 0.0)
        loss = -(target * logps).sum(-1).mean()

        with torch.no_grad():
            probs = logps.exp() * is_option
            accuracy = (probs.argmax(-1) == target.argmax(-1)).float().mean()
            brier = ((probs - target) ** 2).sum(-1).mean()
            self._metrics[mode]["accuracy"].append(self.accelerator.gather_for_metrics(accuracy).mean().item())
            self._metrics[mode]["brier"].append(self.accelerator.gather_for_metrics(brier).mean().item())
        return (loss, option_logits) if return_outputs else loss


tokenizer = AutoTokenizer.from_pretrained(MODEL)
dataset = load_train(num_samples=40_000)
dataset = dataset.map(tokenize, fn_kwargs={"tokenizer": tokenizer}, remove_columns=dataset.column_names)
dataset = dataset.filter(lambda example: len(example["input_ids"]) <= MAX_LENGTH)
# The text model: the vision tower of Qwen3.5 is not needed for text-only questions
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    # The dataset is tokenized and filtered above: preparing it would truncate the prompts at `max_length`, cutting
    # the token the answer is read at
    dataset_kwargs={"skip_prepare_dataset": True},
    # `label_ids` and `target` go through the collator to `compute_loss`
    remove_unused_columns=False,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    num_train_epochs=1,
    logging_steps=25,
    save_strategy="no",
    report_to="trackio",
)
trainer = DecisionTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    data_collator=DecisionCollator(pad_token_id=tokenizer.pad_token_id, padding_free=True),
    peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear"),
)
trainer.train()

# Merge the adapter, so that `transformers serve` loads the model like any other
trainer.model.merge_and_unload().save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
