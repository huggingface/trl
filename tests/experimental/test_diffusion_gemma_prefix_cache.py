# Copyright 2026 The HuggingFace Team. All rights reserved.
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

"""Exact sampled-prefix conditioning with real DiffusionGemma sliding caches."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers import DiffusionGemmaConfig, DiffusionGemmaForBlockDiffusion


Trainer = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "examples/sft_diffusion_gemma/sft_diffusion_gemma.py")
)["DiffusionGemmaSFTTrainer"]


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    config = DiffusionGemmaConfig(
        text_config={
            "vocab_size": 100,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "num_global_key_value_heads": 1,
            "head_dim": 16,
            "global_head_dim": 16,
            "max_position_embeddings": 2048,
            "num_experts": 4,
            "top_k_experts": 2,
            "moe_intermediate_size": 32,
            "layer_types": ["sliding_attention", "full_attention"],
        },
        vision_config={
            "model_type": "gemma4_vision",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "position_embedding_size": 64,
        },
        attn_implementation="eager",
    )
    assert config.text_config.sliding_window == 512
    return DiffusionGemmaForBlockDiffusion(config).train()


@pytest.mark.parametrize("prefix_lengths", [(128, 640), (128, 256)])
def test_prefix_grouped_forward_matches_independent_native_model(tiny_model, prefix_lengths):
    torch.manual_seed(41)
    ids = torch.randint(3, 100, (2, 1024), dtype=torch.long)
    mask = torch.ones_like(ids)
    canvas = torch.randint(3, 100, (2, 256), dtype=torch.long)
    prefix = torch.tensor(prefix_lengths, dtype=torch.long)
    offsets = torch.arange(256, dtype=torch.long)

    grouped = Trainer._prefix_conditioned_logits(tiny_model, ids, mask, canvas, prefix, offsets)
    independent = []
    for i, length in enumerate(prefix_lengths):
        independent.append(
            tiny_model(
                input_ids=ids[i : i + 1, :length],
                attention_mask=mask[i : i + 1, :length],
                decoder_input_ids=canvas[i : i + 1],
                decoder_attention_mask=torch.ones(1, length + 256, dtype=torch.bool),
                decoder_position_ids=prefix[i : i + 1, None] + offsets,
            ).logits
        )
    torch.testing.assert_close(grouped, torch.cat(independent, dim=0), atol=1e-6, rtol=1e-5)

    # Ensure gradients from *both* separate prefix encodings flow through
    # the shared model parameters and are not discarded by regrouping.
    tiny_model.zero_grad(set_to_none=True)
    grouped.sum().backward()
    grads_grouped = [p.grad.detach().clone() for p in tiny_model.parameters() if p.grad is not None]
    tiny_model.zero_grad(set_to_none=True)
    torch.cat(independent, dim=0).sum().backward()
    grads_independent = [p.grad.detach().clone() for p in tiny_model.parameters() if p.grad is not None]
    assert len(grads_grouped) == len(grads_independent) > 0
    for grad, reference in zip(grads_grouped, grads_independent, strict=True):
        torch.testing.assert_close(grad, reference, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("length", [384, 1024])
def test_full_autoregressive_coloss_and_prefix_decoder_real_cpu(tiny_model, length):
    torch.manual_seed(23)
    ids = torch.randint(3, 100, (1, length))
    labels = ids.clone()
    labels[:, :128] = -100
    mask = torch.ones_like(ids)
    trainer = SimpleNamespace(
        args=SimpleNamespace(max_length=length),
        model=tiny_model,
        canvas_length=256,
        vocab_size=tiny_model.config.text_config.vocab_size,
        final_logit_softcapping=tiny_model.config.text_config.final_logit_softcapping,
        eos_token_id=1,
        model_prediction_type="mean",
        eps=Trainer.eps,
        self_conditioning_p=Trainer.self_conditioning_p,
        _prefix_conditioned_logits=Trainer._prefix_conditioned_logits,
    )
    cached_shapes = []

    def trace_decoder(module, args, kwargs):
        cache = kwargs["past_key_values"]
        cached_shapes.append((int(kwargs["decoder_position_ids"][0, 0]), int(cache.get_mask_sizes(256, 0)[1])))

    handle = tiny_model.model.decoder.register_forward_pre_hook(trace_decoder, with_kwargs=True)
    try:
        torch.manual_seed(3)
        loss, output = Trainer.compute_loss(
            trainer,
            tiny_model,
            {"input_ids": ids, "attention_mask": mask, "labels": labels},
            return_outputs=True,
        )
    finally:
        handle.remove()

    assert bool(torch.isfinite(loss))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in tiny_model.parameters() if p.grad is not None)

    with torch.no_grad():
        expected_encoder = tiny_model.model.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
    torch.testing.assert_close(output.encoder_last_hidden_state, expected_encoder, atol=1e-5, rtol=1e-5)
    assert output.past_key_values is None

    if length == 1024:
        # First model call supplies the full encoder co-loss; subsequent calls
        # build the cache with the selected prefix, without evicting that prefix.
        assert len(cached_shapes) == 3
        assert cached_shapes[0][1] > 0
        assert cached_shapes[1][1] == 0
        assert cached_shapes[2][1] == 0


def test_full_sequence_cache_eviction_changes_native_diffusion_gradients(tiny_model):
    # The failure is not a tensor-shape problem: a decoder mask cannot
    # resurrect prefix keys already evicted by a native sliding cache.
    torch.manual_seed(23)
    ids = torch.randint(3, 100, (1, 1024))
    mask = torch.ones_like(ids)
    canvas = torch.randint(3, 100, (1, 256))
    targets = torch.randint(3, 100, (1, 256))
    start = torch.tensor([128])
    offsets = torch.arange(256)

    cache_offsets = []

    def trace(module, args, kwargs):
        cache_offsets.append(int(kwargs["past_key_values"].get_mask_sizes(256, 0)[1]))

    hook = tiny_model.model.decoder.register_forward_pre_hook(trace, with_kwargs=True)
    try:
        old_logits = tiny_model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=canvas,
            decoder_attention_mask=torch.cat(
                [torch.arange(1024)[None] < 128, torch.ones(1, 256, dtype=torch.bool)], dim=1
            ),
            decoder_position_ids=start[:, None] + offsets,
        ).logits
        new_logits = Trainer._prefix_conditioned_logits(tiny_model, ids, mask, canvas, start, offsets)
    finally:
        hook.remove()

    assert len(cache_offsets) == 2
    assert cache_offsets[0] > 128  # long encoder has evicted the selected prefix
    assert cache_offsets[1] == 0  # exact selected-prefix encoder keeps all 128 tokens
    assert float((old_logits - new_logits).abs().max()) > 1e-5

    old_loss = F.cross_entropy(old_logits.flatten(0, 1), targets.flatten())
    new_loss = F.cross_entropy(new_logits.flatten(0, 1), targets.flatten())
    old_grad = torch.autograd.grad(old_loss, tiny_model.lm_head.weight)[0]
    new_grad = torch.autograd.grad(new_loss, tiny_model.lm_head.weight)[0]
    assert bool(torch.isfinite(new_grad).all())
    assert float((old_grad - new_grad).abs().max()) > 1e-6
