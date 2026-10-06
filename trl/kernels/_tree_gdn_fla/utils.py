# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in this directory.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl
from packaging.version import Version


IS_NVIDIA_HOPPER = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9
IS_NVIDIA_BLACKWELL = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10
TRITON_ABOVE_3_4_0 = Version(triton.__version__) >= Version("3.4.0")
TRITON_ABOVE_3_7_1 = Version(triton.__version__) >= Version("3.7.1")


def check_shared_mem(arch="ampere", device=None):
    if not torch.cuda.is_available():
        return False
    required = {"ampere": 163840, "ada": 101376, "hopper": 232448}[arch]
    return torch.cuda.get_device_properties(device).shared_memory_per_block_optin >= required


@triton.jit
def exp2(x):
    return tl.math.exp2(x.to(tl.float32))
