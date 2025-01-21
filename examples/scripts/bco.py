# Copyright 2025 The HuggingFace Team. All rights reserved.
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

"""
Run the BCO training script with the commands below. In general, the optimal configuration for BCO will be similar to that of KTO.

# Full training:
python examples/scripts/bco.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --trust_remote_code \
    --dataset_name trl-lib/ultrafeedback-gpt-3.5-turbo-helpfulness \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs 1 \
    --learning_rate 1e-6 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --logging_steps 0.01 \
    --eval_steps 0.2 \
    --save_strategy no \
    --output_dir=bco-aligned-model \
    --logging_first_step \
    --max_length 2048 \
    --max_prompt_length 1536 \
    --max_completion_length 1024 \
    --no_remove_unused_columns \
    --warmup_ratio 0.1 \
    --bf16 \
    --report_to wandb

# QLoRA:
python examples/scripts/bco.py \
    --model_name_or_path=nnheui/stablelm-2-1_6b-sft-full \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs 1 \
    --learning_rate 1e-6 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --logging_steps 0.01 \
    --eval_steps 0.2 \
    --save_strategy no \
    --output_dir=bco-aligned-model-lora \
    --logging_first_step \
    --warmup_ratio 0.1 \
    --report_to wandb \
    --max_length 2048 \
    --max_prompt_length 1536 \
    --max_completion_length 1024 \
    --no_remove_unused_columns \
    --warmup_ratio 0.1 \
    --bf16 \
    --use_peft \
    --load_in_4bit \
    --lora_target_modules=all-linear \
    --lora_r=16 \
    --lora_alpha=16
"""

from functools import partial

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import load_dataset
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, PreTrainedModel

from trl import BCOConfig, BCOTrainer, ModelConfig, ScriptArguments, get_peft_config, setup_chat_format


def embed_prompt(input_ids: torch.LongTensor, attention_mask: torch.LongTensor, model: PreTrainedModel):
    """
    Borrowed from https://huggingface.co/nomic-ai/nomic-embed-text-v1.5#transformers
    """

    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    with torch.no_grad():
        model_output = model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = mean_pooling(model_output, attention_mask)

    matryoshka_dim = 512
    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
    embeddings = embeddings[:, :matryoshka_dim]

    return embeddings


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, BCOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()

    training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    # Load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # If we are aligning a base model, we use ChatML as the default template
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    accelerator = Accelerator()
    embedding_model = AutoModel.from_pretrained(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=model_args.trust_remote_code,
        safe_serialization=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    embedding_model = accelerator.prepare_model(embedding_model)
    embedding_tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased", trust_remote_code=model_args.trust_remote_code
    )
    embedding_func = partial(
        embed_prompt,
        model=embedding_model,
    )

    # Initialize the BCO trainer
    trainer = BCOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        embedding_func=embedding_func,
        embedding_tokenizer=embedding_tokenizer,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
