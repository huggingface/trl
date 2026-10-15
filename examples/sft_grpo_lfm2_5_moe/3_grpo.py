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
#     "trl[vllm,peft]",
#     "trackio",
# ]
# ///

# docstyle-ignore
"""
GRPO of LiquidAI/LFM2.5-8B-A1B as a research agent on HotpotQA: each question comes with a library of ten Wikipedia
articles, two of which hold the answer, and the model can only reach them through `search` and `read` tools before it
calls `submit_answer`. The reward is the exact match of the submitted answer, so the model is only paid for finding the
right articles and chaining them.

On one GPU, with vLLM colocated:

    python examples/sft_grpo_lfm2_5_moe/3_grpo.py

On one 8-GPU node (each GPU runs its own vLLM engine and trains on its own slice of the batch):

    accelerate launch --num_processes 8 examples/sft_grpo_lfm2_5_moe/3_grpo.py

Evaluate the base model and the trained adapter on the same held-out questions with `2_evaluate.py`.
"""

from peft import LoraConfig
from wiki_library import WikiLibrary, load_questions

from trl import GRPOConfig, GRPOTrainer


MODEL = "LiquidAI/LFM2.5-8B-A1B"

training_args = GRPOConfig(
    output_dir="LFM2.5-8B-A1B-GRPO-HotpotQA",
    model_init_kwargs={"dtype": "bfloat16"},
    # LoRA trains ~1% of the parameters, so the usual GRPO learning rate is far too small.
    learning_rate=1e-4,
    use_vllm=True,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.35,
    vllm_max_model_length=6144,
    # 4 questions x 8 episodes per step
    per_device_train_batch_size=4,
    num_generations=8,
    gradient_accumulation_steps=8,
    # The whole episode (every turn, tool results included) has to fit in the completion
    max_completion_length=4096,
    max_tool_calling_iterations=10,
    # vLLM and the trainer disagree by ~0.035 nats per generated token on this model, so a sequence-level ratio (the
    # default) multiplies that over a 1-2k-token episode and down-weights most episodes by orders of magnitude.
    # Correct each token instead.
    vllm_importance_sampling_mode="token_truncate",
    max_steps=200,
    logging_steps=1,
    log_completions=True,
    save_steps=50,
    report_to="trackio",
)
trainer = GRPOTrainer(
    model=MODEL,
    args=training_args,
    train_dataset=load_questions("train"),
    environment_factory=WikiLibrary,
    peft_config=LoraConfig(r=32, lora_alpha=64, target_modules="all-linear"),
)
trainer.train()
