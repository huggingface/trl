# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0

import functools
import inspect
import types
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from trl.trainer.sft_trainer import _patch_chunked_ce_lm_head


class _TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MagicMock()
        self.config.text_config = None
        # top-level config attrs used when is_vlm=False
        self.config.final_logit_softcapping = None
        self.config.logit_scale = 1.0
        self.config.output_router_logits = False
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.base_model = MagicMock()

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        return MagicMock(loss=torch.tensor(0.0), logits=None)


def test_patch_chunked_ce_accepts_partial_forward():
    """Qwen3.5-style models expose forward as functools.partial (#6483)."""
    model = _TinyLM()
    real = model.forward
    model.forward = functools.partial(real)  # no __func__

    # Should not raise AttributeError: 'partial' object has no attribute '__func__'
    _patch_chunked_ce_lm_head(model, chunk_size=2, is_vlm=False)

    assert isinstance(model.forward, types.MethodType)
    sig = inspect.signature(model.forward)
    # Bound method signature should include original kwargs surface
    assert "input_ids" in sig.parameters or len(sig.parameters) >= 1
