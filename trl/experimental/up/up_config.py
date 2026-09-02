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

from dataclasses import dataclass

from ...trainer.grpo_config import GRPOConfig


@dataclass
class UPConfig(GRPOConfig):
    # docstyle-ignore
    r"""
    Configuration class for the [`UPTrainer`].

    [`UPConfig`] inherits every parameter from [`GRPOConfig`] and adds none. It exists to change which parameters are
    meaningful: in UP, tokens with positive advantage bypass clipping entirely, so only the lower clip bound is ever
    consulted.

    - `loss_type` does not select the objective: [`UPTrainer`] always uses the UP loss with DAPO's global
      active-token normalization, matching the paper's UP-DAPO instantiation. It is still read on inherited code
      paths (validation and warnings in [`GRPOTrainer`]), so the default `"dapo"` is kept, since that is the
      aggregation actually used.
    - `epsilon` (and `delta`, if set) still apply, but only to the non-positive-advantage branch.
    - `epsilon_high` has **no effect** with the default `delta=None`, and more generally whenever
      `delta >= 1 + epsilon_high`: positive advantages skip clipping, and for non-positive ones the upper bound is
      dominated. Only a `delta` set below `1 + epsilon_high` makes it bind. Table A1 of the [UP
      paper](https://huggingface.co/papers/2607.06987) lists no `ε_high` for UP-DAPO accordingly.

    The paper's UP-DAPO setup uses `beta=0.0` at 14B. At smaller scale, consider keeping a KL term (`beta > 0`), as
    the paper's UP-GRPO variant does.
    """
