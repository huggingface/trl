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
#     "transformers>=5.23.0",  # `transformers.decision`
#     "trackio",
# ]
# ///

# docstyle-ignore
"""
Trains Qwen/Qwen3.5-4B as a decision ("System One") model: it answers typed questions about a state with a probability
for each option, from the next-token logits of the option labels (`Yes`/`No`, `A`, `B`, ...) at the end of the prompt.
The prompt is built by `transformers.decision`, like the `/v1/systemone` endpoint of `transformers serve` does for a
model without a decision config, so the merged model is served as is. LoRA on 40k questions of
tasksource-jev-typed-decisions, with their soft targets. On one L4 (24 GB): 961 steps of 32 questions, 6.8 s/step,
10.6 GB peak; one epoch takes 108 minutes.

    python examples/sft_decision_model/1_sft.py
"""

import torch
from decisions import load_train
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.decision import DecisionConfig, label_token_ids

from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
from trl.trainer.utils import pad


MODEL = "Qwen/Qwen3.5-4B"
OUTPUT_DIR = "Qwen3.5-4B-Decision"
MAX_LENGTH = 512
# The default config, which `transformers serve` uses for a model without a `decision.json`
DECISION_CONFIG = DecisionConfig()


def tokenize(example, tokenizer):
    # One prompt for the question, read after the generation prompt: at its last token
    request = {"state": example["state"], "questions": {"q": example["question"]}}
    [(inputs, [question])] = DECISION_CONFIG.prepare_inputs(request, tokenizer)
    return {
        "input_ids": inputs["input_ids"][0].tolist(),
        "label_ids": label_token_ids(tokenizer, tuple(question.labels)),
        # In the order of the options of the prompt
        "target": [example["target"][option] for option in question.options],
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
        # Padding-free: the prompts are one sequence, and only the logits at the last token of each are computed (the
        # sequence boundaries are padded with empty sequences after the prompts)
        last_tokens = inputs["cu_seq_lens_q"][1 : len(label_ids) + 1] - 1
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
# The tokenizer as loaded: `prepare_inputs` added its sentinel tokens to the one above, which serving adds again
AutoTokenizer.from_pretrained(MODEL).save_pretrained(OUTPUT_DIR)
