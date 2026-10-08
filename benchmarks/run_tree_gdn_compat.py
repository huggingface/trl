"""Rerun current tree tests against Transformers 5.8.1 without editing the working tree."""

import runpy
import sys

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer


original_init = Qwen3_5DecoderLayer.__init__


def init_with_block_type(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self.block_type = self.layer_type


Qwen3_5DecoderLayer.__init__ = init_with_block_type
original_assert_close = torch.testing.assert_close


def assert_close_with_error_norm(actual, expected, *args, **kwargs):
    try:
        return original_assert_close(actual, expected, *args, **kwargs)
    except AssertionError as error:
        if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
            difference = actual.float() - expected.float()
            relative = difference.norm() / expected.float().norm().clamp_min(1e-8)
            raise AssertionError(
                f"{error}\nrelative_l2={relative.item():.9g}, max_abs={difference.abs().max().item():.9g}, "
                f"reference_norm={expected.float().norm().item():.9g}"
            ) from error
        raise


torch.testing.assert_close = assert_close_with_error_norm
if sys.argv[1] == "--match-fused-conv":
    sys.argv.pop(1)
    from trl.experimental.async_grpo.tree import gdn

    original_conv = gdn.tree_causal_conv1d

    def matched_conv(*args, **kwargs):
        kwargs["activation_in_fp32"] = True
        return original_conv(*args, **kwargs)

    gdn.tree_causal_conv1d = matched_conv
    print("Diagnostic only: tree convolution uses FP32 SiLU to match causal-conv1d", flush=True)  # noqa: T201
target = sys.argv.pop(1)
print("Test-only compatibility alias: block_type = layer_type (Transformers 5.8.1)", flush=True)  # noqa: T201
if target == "pytest":
    import pytest

    raise SystemExit(pytest.main(sys.argv[1:]))
sys.argv[0] = target
runpy.run_path(target, run_name="__main__")
