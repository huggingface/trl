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

import torch


def tree_path_sources(cu_seqlens, parents, n_tokens, width):
    """
    `(n_tokens, width)` indices: column `k` holds the token `k` steps back along each token's own root path, into its
    parent segments, or `-1` once the walk passes the root. This is the window a causal convolution reads on a tree.

    Example: segments `[0, 1]` (a root), `[2, 3]` and `[4]` (children of the first) give a second column `[-1, 0, 1, 2,
    1]`.
    """
    parents = parents.long()
    tokens = torch.arange(n_tokens, device=cu_seqlens.device)
    segment = torch.arange(parents.numel(), device=tokens.device).repeat_interleave(
        cu_seqlens[1:] - cu_seqlens[:-1], output_size=n_tokens
    )
    previous = torch.where(tokens == cu_seqlens[segment], cu_seqlens[parents + 1][segment] - 1, tokens - 1)
    sources = [tokens]
    for _ in range(width - 1):
        sources.append(torch.where(sources[-1] >= 0, previous[sources[-1].clamp(min=0)], -1))
    return torch.stack(sources, dim=1)
