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

import pytest
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen3MoeConfig, Qwen3MoeForCausalLM

from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import _get_expert_usage_counts, _summarize_expert_usage


def test_expert_usage_counts_use_top_k_and_ignore_padding():
    router_logits = (
        torch.tensor(
            [
                [[9.0, 8.0, 1.0, 0.0], [1.0, 9.0, 8.0, 0.0], [9.0, 1.0, 0.0, 8.0]],
                [[0.0, 8.0, 9.0, 1.0], [0.0, 1.0, 8.0, 9.0], [9.0, 8.0, 1.0, 0.0]],
            ]
        ),
        torch.tensor(
            [
                [[1.0, 9.0, 8.0, 0.0], [9.0, 1.0, 8.0, 0.0], [8.0, 9.0, 1.0, 0.0]],
                [[8.0, 0.0, 9.0, 1.0], [0.0, 9.0, 1.0, 8.0], [9.0, 1.0, 8.0, 0.0]],
            ]
        ),
    )
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 0]])

    counts = _get_expert_usage_counts(router_logits, num_experts_per_tok=2, attention_mask=attention_mask)

    torch.testing.assert_close(counts, torch.tensor([[1, 3, 3, 1], [2, 2, 3, 1]]))
    assert counts.sum().item() == 2 * 4 * 2  # layers * valid tokens * selected experts


def test_expert_usage_counts_reject_mask_shape_mismatch():
    router_logits = (torch.randn(5, 4),)

    with pytest.raises(ValueError, match="5 token rows"):
        _get_expert_usage_counts(router_logits, num_experts_per_tok=2, attention_mask=torch.ones(2, 3))


def test_expert_usage_summary():
    counts = torch.tensor([[1, 1, 1, 1], [0, 0, 2, 2]])

    metrics = _summarize_expert_usage(counts)

    assert metrics["expert_usage/normalized_entropy_mean"] == pytest.approx(0.75)
    assert metrics["expert_usage/normalized_entropy_min"] == pytest.approx(0.5)
    assert metrics["expert_usage/max_share_mean"] == pytest.approx(0.375)
    assert metrics["expert_usage/max_share_max"] == pytest.approx(0.5)
    assert metrics["expert_usage/active_fraction_mean"] == pytest.approx(0.75)


def test_expert_usage_is_opt_in(tmp_path):
    args = SFTConfig(output_dir=str(tmp_path), bf16=False)

    assert args.log_expert_usage is False


def test_expert_usage_rejects_liger(tmp_path):
    with pytest.raises(ValueError, match="not currently supported"):
        SFTConfig(output_dir=str(tmp_path), bf16=False, log_expert_usage=True, use_liger_kernel=True)


@pytest.mark.parametrize("loss_type", ["nll", "chunked_nll"])
def test_sft_trainer_logs_bounded_expert_usage_metrics(tmp_path, loss_type):
    backend_tokenizer = Tokenizer(
        WordLevel(
            {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, "monitor": 3, "moe": 4, "routing": 5, "padding": 6},
            unk_token="[UNK]",
        )
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend_tokenizer, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]"
    )
    model = Qwen3MoeForCausalLM(
        Qwen3MoeConfig(
            vocab_size=len(tokenizer),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            moe_intermediate_size=16,
            num_experts_per_tok=2,
            num_experts=4,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    )
    dataset = Dataset.from_dict(
        {
            "input_ids": [[3, 4, 5, 2], [6, 5, 4, 2]],
            "labels": [[3, 4, 5, 2], [6, 5, 4, 2]],
        }
    )
    args = SFTConfig(
        output_dir=str(tmp_path),
        per_device_eval_batch_size=2,
        max_length=16,
        loss_type=loss_type,
        log_expert_usage=True,
        router_aux_loss_coef=0.0,
        gradient_checkpointing=False,
        dataloader_pin_memory=False,
        bf16=False,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=dataset,
        processing_class=tokenizer,
    )

    metrics = trainer.evaluate()

    expert_usage_metrics = {key for key in metrics if key.startswith("eval_expert_usage/")}
    assert expert_usage_metrics == {
        "eval_expert_usage/normalized_entropy_mean",
        "eval_expert_usage/normalized_entropy_min",
        "eval_expert_usage/max_share_mean",
        "eval_expert_usage/max_share_max",
        "eval_expert_usage/active_fraction_mean",
    }
