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

import torch
from examples.xtoken.sort_and_cut_projection_matrix import sort_and_cut


def test_sort_and_cut_preserves_scale_slot_with_sentinel_index(tmp_path):
    input_path = tmp_path / "input.pt"
    output_path = tmp_path / "output.pt"
    torch.save(
        {
            "indices": torch.tensor([[3, 4, -1], [5, -1, -1]]),
            "likelihoods": torch.tensor([[0.5, 0.3, 0.2], [0.8, 0.0, 0.2]]),
            "enable_scale_trick": True,
        },
        input_path,
    )

    sort_and_cut(input_path, output_path, new_top_k=2, preserve_last=True, verbose=False)
    output = torch.load(output_path, weights_only=False)

    torch.testing.assert_close(output["indices"][:, -1], torch.tensor([-1, -1]))
    assert (output["likelihoods"][:, -1] > 0).all()
