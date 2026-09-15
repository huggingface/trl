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

"""
Two-rank CPU check that the multi-teacher loss issues the same collectives on every rank when the student head is a
sharded `DTensor` (what FSDP2 gives it), even though local routing puts a different number of teacher groups on each
rank. See `distillation_dtensor_head_script.py` for what the worker builds and why the head is built by hand.

Launched with `python -m torch.distributed.run` for the same reason as
`test_distillation_trainer_multi_teacher.py`: this environment has no MPI, so `accelerate launch` cannot spawn a
`MULTI_CPU` group.
"""

import os
import subprocess
import sys

import pytest

from ..testing_utils import TrlTestCase


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "distillation_dtensor_head_script.py")


@pytest.mark.slow
class TestDistillationTrainerDTensorHeadTwoRankCpu(TrlTestCase):
    @pytest.mark.parametrize("equal_groups", [False, True])
    def test_sharded_student_head_loss_runs_with_unequal_teacher_groups(self, equal_groups):
        environment = dict(
            os.environ,
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            TOKENIZERS_PARALLELISM="false",
            # Turns a rank-dependent number of collectives into a reported mismatch instead of a hang.
            TORCH_DISTRIBUTED_DEBUG="DETAIL",
        )
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            SCRIPT,
            *(["--equal-groups"] if equal_groups else []),
        ]
        # `torchrun` fails the whole run if any rank raises, so a zero exit code is both ranks reaching the end.
        result = subprocess.run(command, capture_output=True, text=True, timeout=1800, env=environment, cwd=ROOT)
        assert result.returncode == 0, f"run failed:\n{result.stdout[-4000:]}\n{result.stderr[-8000:]}"
