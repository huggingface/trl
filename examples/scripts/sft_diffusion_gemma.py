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
# ]
# ///

"""
Supervised fine-tuning of DiffusionGemma, a block-diffusion language model, on GSM8K with LoRA.

DiffusionGemma ([`google/diffusiongemma-26B-A4B-it`](https://huggingface.co/google/diffusiongemma-26B-A4B-it), requires
`transformers >= 5.11`) couples a causal encoder with a bidirectional decoder: the encoder reads clean context tokens
into a KV cache, and the decoder denoises a block of response tokens (the "canvas") with bidirectional attention,
cross-attending to that cache. Training therefore differs from autoregressive SFT:

1. Per example, one response block of `canvas_length` tokens is selected at random; the encoder reads the full clean
   sequence, and the decoder mask restricts the canvas to the prompt plus the clean response blocks before it.
2. The clean canvas (EOS-filled past the end of the response) is corrupted by independently replacing tokens with
   uniform-random vocabulary tokens with per-example probability `t ~ U(eps, 1)`. There is no mask token.
3. With probability 0.5 per example, the decoder is self-conditioned on its own logits from a first, no-grad pass.
4. The loss is plain mean cross-entropy between the decoder logits and the clean tokens over the whole canvas
   (corrupted and uncorrupted positions alike), plus an autoregressive co-loss on the encoder.

Requires transformers with https://github.com/huggingface/transformers/pull/46568.

The script trains with `assistant_only_loss`, so [`SFTTrainer`] swaps in TRL's DiffusionGemma training chat template
(`trl/chat_templates/diffusion_gemma_training.jinja`), whose `{% generation %}` markers cover the assistant content
plus the closing `<turn|>` token.

LoRA follows the reference fine-tune of the released checkpoint: rank 16, alpha 32, adapting the attention and
dense-MLP linears of the encoder and decoder layers only; the MoE experts, the router, the self-conditioning block,
and the vision tower stay frozen.

Usage:

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft_diffusion_gemma.py \
    --model_name_or_path google/diffusiongemma-26B-A4B-it \
    --dtype bfloat16 \
    --attn_implementation sdpa \
    --dataset_name openai/gsm8k \
    --dataset_config main \
    --use_peft \
    --learning_rate 1.5e-4 \
    --adam_beta1 0.95 \
    --adam_beta2 0.99 \
    --weight_decay 1e-4 \
    --warmup_steps 25 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr": 1.5e-5}' \
    --max_steps 800 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_length 1024 \
    --output_dir diffusiongemma-26B-A4B-it-gsm8k-lora
"""

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, DiffusionGemmaForBlockDiffusion

from trl import ModelConfig, ScriptArguments, SFTConfig, SFTTrainer, TrlParser


class DiffusionGemmaSFTTrainer(SFTTrainer):
    """
    SFTTrainer with the block-diffusion objective of DiffusionGemma.

    Expects the batches produced by the default SFT collator (`input_ids`, `attention_mask`, and `labels` with `-100`
    outside the assistant response) and replaces the autoregressive loss with the denoising loss described in the
    module docstring.
    """

    eps = 1e-3  # minimum corruption ratio, avoids zero-corruption samples
    self_conditioning_p = 0.5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = self.model.config
        self.canvas_length = config.canvas_length
        self.vocab_size = config.text_config.vocab_size
        self.final_logit_softcapping = config.text_config.final_logit_softcapping
        self.eos_token_id = self.processing_class.eos_token_id

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        block_size = self.canvas_length

        # The completion is the contiguous suffix of supervised tokens
        supervised = labels != -100
        prefix_len = supervised.int().argmax(dim=1)
        response_len = supervised.sum(dim=1)

        # One canvas per step: select one response block per example. The encoder reads the full clean sequence (its
        # autoregressive co-loss covers it all), but the decoder may only see the prompt and the clean response
        # blocks *before* the selected one, so the decoder mask cuts the cache off there.
        num_blocks = (response_len - 1).clamp(min=0) // block_size + 1
        block_idx = (torch.rand(batch_size, device=device) * num_blocks).long()
        encoder_len = prefix_len + block_idx * block_size

        # Clean canvas: the selected block, EOS-filled up to block_size past the end of the response. The fill is
        # supervised, so every canvas position contributes to the loss.
        offsets = torch.arange(block_size, device=device)
        abs_idx = (encoder_len[:, None] + offsets).clamp(max=seq_len - 1)
        in_response = offsets < (response_len - block_idx * block_size)[:, None]
        canvas_target = torch.where(in_response, input_ids.gather(1, abs_idx), self.eos_token_id)

        # Uniform random-token corruption: per-example t ~ U(eps, 1), each canvas position is independently replaced
        # with a token drawn uniformly from the vocabulary (there is no mask token)
        t = self.eps + (1 - self.eps) * torch.rand(batch_size, 1, device=device)
        corrupt = torch.rand(batch_size, block_size, device=device) < t
        random_tokens = torch.randint(self.vocab_size, (batch_size, block_size), device=device)
        canvas_ids = torch.where(corrupt, random_tokens, canvas_target)

        cache_mask = (torch.arange(seq_len, device=device) < encoder_len[:, None]).long()
        canvas_mask = torch.ones(batch_size, block_size, dtype=torch.long, device=device)
        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": canvas_ids,
            "decoder_attention_mask": torch.cat([cache_mask, canvas_mask], dim=1),
            "decoder_position_ids": encoder_len[:, None] + offsets,
        }

        # Two-pass self-conditioning: a first no-grad pass produces the logits the second pass conditions on,
        # gated per example with probability p
        with torch.no_grad():
            model_kwargs["self_conditioning_logits"] = model(**model_kwargs).logits
        model_kwargs["self_conditioning_mask"] = torch.rand(batch_size, device=device) < self.self_conditioning_p

        outputs = model(**model_kwargs)

        # Flat cross-entropy over all canvas positions, corrupted and uncorrupted (no 1/t weighting: the uniform
        # corruption kernel has a flat ELBO)
        diffusion_loss = F.cross_entropy(outputs.logits.flatten(0, 1), canvas_target.flatten())

        # Autoregressive co-loss on the encoder, over all valid next-token pairs
        encoder_logits = self.model.lm_head(outputs.encoder_last_hidden_state).float()
        cap = self.final_logit_softcapping
        encoder_logits = torch.tanh(encoder_logits / cap) * cap
        ar_mask = attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()
        ar_targets = torch.where(ar_mask, input_ids[:, 1:], -100)
        ar_loss = F.cross_entropy(encoder_logits[:, :-1].flatten(0, 1), ar_targets.flatten(), ignore_index=-100)

        loss = diffusion_loss + ar_loss
        return (loss, outputs) if return_outputs else loss


def main(script_args, training_args, model_args):
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        model_args.model_name_or_path,
        dtype=model_args.dtype,
        attn_implementation=model_args.attn_implementation,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    if model_args.use_peft:
        # LoRA on the attention and dense-MLP linears of the encoder/decoder layers only; the MoE experts, the
        # router, the self-conditioning block, and the vision tower stay frozen. The encoder and decoder share
        # tied base weights but get independent adapters, so keep the adapters unmerged (`merge_and_unload` would
        # fold both deltas into the shared tensors).
        peft_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=r"model\.(encoder\.language_model|decoder)\.layers\.\d+"
            r"\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)",
        )
    else:
        peft_config = None

    dataset = load_dataset(script_args.dataset_name, script_args.dataset_config, split=script_args.dataset_train_split)

    def to_messages(example):
        return {
            "messages": [
                {"role": "user", "content": example["question"]},
                {"role": "assistant", "content": example["answer"]},
            ]
        }

    dataset = dataset.map(to_messages, remove_columns=dataset.column_names)

    # The supervised canvas is derived from the labels, so the loss must be restricted to the assistant response.
    # This also makes SFTTrainer swap in TRL's DiffusionGemma training chat template.
    training_args.assistant_only_loss = True
    # The decoder reads the encoder's KV cache, but gradient checkpointing drops cache writes during training,
    # leaving the decoder without context.
    training_args.gradient_checkpointing = False

    trainer = DiffusionGemmaSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
