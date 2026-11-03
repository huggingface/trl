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
#     "trl>=1.18.0",
#     "transformers>=5.23.0",
#     "kernels>=0.17.2",
#     "trackio",
# ]
# ///

# docstyle-ignore
"""
Full fine-tuning of gpt-oss-20b on 500k chess puzzles, then back to MXFP4. TRL loads the MXFP4 experts dequantized,
FSDP2 shards the model across the GPUs of one node, and the fine-tuned experts are quantized again at the end, so the
model is served like the original. About 2 hours on 4xH200; see README.md for the results.

    accelerate launch --config_file examples/accelerate_configs/fsdp2.yaml --num_processes 4 \
        examples/sft_gpt_oss_chess/sft_gpt_oss_chess.py
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, Mxfp4Config

from trl import SFTConfig, SFTTrainer


MODEL = "openai/gpt-oss-20b"

# Each prompt shows the board, its FEN and the legal moves; the completion is the first move of the solution.
train_dataset = load_dataset("trl-lib/chess-puzzles", split="train[:500000]")

training_args = SFTConfig(
    output_dir="gpt-oss-20b-chess",
    model_init_kwargs={
        # Flash Attention 3 from the Hub runs on Hopper and keeps packed sequences apart.
        "attn_implementation": "kernels-community/vllm-flash-attn3",
    },
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    max_length=2048,
    packing=True,
    learning_rate=2e-5,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr_rate": 0.1},
    warmup_steps=0.03,
    logging_steps=1,
    save_strategy="no",
    report_to="trackio",
)
trainer = SFTTrainer(model=MODEL, args=training_args, train_dataset=train_dataset)
trainer.train()
trainer.save_model()

# Loading the fine-tuned model with `Mxfp4Config()` quantizes its experts to MXFP4, like the original checkpoint.
if trainer.accelerator.is_main_process:
    model = AutoModelForCausalLM.from_pretrained(
        training_args.output_dir, quantization_config=Mxfp4Config(), dtype=torch.bfloat16, device_map="cuda"
    )
    model.save_pretrained(f"{training_args.output_dir}-mxfp4")
    trainer.processing_class.save_pretrained(f"{training_args.output_dir}-mxfp4")

trainer.end()
