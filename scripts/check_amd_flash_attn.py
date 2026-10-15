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

"""Check the AITER Hub kernel against PyTorch math attention on each visible ROCm GPU."""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from kernels import get_kernel


def main():
    assert torch.version.hip, "This check requires ROCm torch"
    assert torch.cuda.is_available(), "ROCm GPU is unavailable"
    # ROCm may report zero devices before HIP is initialized.
    torch.cuda.init()
    kernel = get_kernel("kernels-community/aiter-flash-attn", version=1)

    for device_index in range(torch.cuda.device_count()):
        with torch.cuda.device(device_index):
            torch.manual_seed(42)
            for dtype in (torch.float16, torch.bfloat16):
                for causal in (False, True):
                    # Flash Attention uses (batch, sequence, heads, head_dim).
                    q, k, v = [
                        torch.randn(2, 128, 4, 64, device=f"cuda:{device_index}", dtype=dtype, requires_grad=True)
                        for _ in range(3)
                    ]
                    output = kernel.flash_attn_func(q, k, v, causal=causal)
                    grad_output = torch.randn_like(output)
                    grads = torch.autograd.grad(output, (q, k, v), grad_output)

                    # Use float32 math attention as an independent forward/backward reference.
                    q_ref, k_ref, v_ref = [x.detach().float().requires_grad_() for x in (q, k, v)]
                    with sdpa_kernel(SDPBackend.MATH):
                        reference = F.scaled_dot_product_attention(
                            q_ref.transpose(1, 2), k_ref.transpose(1, 2), v_ref.transpose(1, 2), is_causal=causal
                        ).transpose(1, 2)
                    ref_grads = torch.autograd.grad(reference, (q_ref, k_ref, v_ref), grad_output.float())

                    tolerance = 2e-2 if dtype == torch.bfloat16 else 3e-3
                    for actual, expected in zip((output, *grads), (reference, *ref_grads), strict=True):
                        assert torch.isfinite(actual).all(), "AITER produced non-finite outputs or gradients"
                        torch.testing.assert_close(actual.float(), expected, atol=tolerance, rtol=tolerance)
                    print(f"AITER Flash Attention passed: GPU {device_index}, {dtype}, causal={causal}")


if __name__ == "__main__":
    main()
