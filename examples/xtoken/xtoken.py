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
#     "trl @ git+https://github.com/huggingface/trl.git",
#     "trackio",
# ]
# ///

# docstyle-ignore
"""
python examples/xtoken/xtoken.py \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --teacher_model_name_or_path Qwen/Qwen3-4B \
    --dataset_name trl-lib/chatbot_arena_completions \
    --xtoken_loss_type p_kl \
    --xtoken_projection_matrix_path cross_tokenizer_data/projection_map_Llama-3.2-1B-Instruct_to_Qwen3-4B_multitoken_top_32_double_top4.pt \
    --lmbda 0.0 \
    --max_steps 5000 \
    --output_dir xtoken-output

Build the projection matrix first with the scripts in examples/xtoken/. See
https://huggingface.co/papers/2605.21699.
"""

from datasets import load_dataset
from transformers import AutoTokenizer

from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config, get_quantization_config
from trl.experimental.gold import GOLDConfig, GOLDTrainer


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    ################
    # Model & Tokenizer
    ################
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs
    training_args.teacher_model_init_kwargs = dict(
        revision=training_args.teacher_model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
        use_cache=True,
        quantization_config=quantization_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    dataset = load_dataset(
        script_args.dataset_name,
        name=script_args.dataset_config,
        split=script_args.dataset_train_split,
        streaming=script_args.dataset_streaming,
    )

    ################
    # Training
    ################
    trainer = GOLDTrainer(
        model=model_args.model_name_or_path,
        teacher_model=training_args.teacher_model_name_or_path,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )
    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
