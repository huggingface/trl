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
Two-rank CPU check of managed multi-teacher distillation (the `teacher_models` constructor argument): two gloo
processes on CPU, compared against a single-process reference trained on the same global batch and the same tokens
(see `distillation_multi_teacher_script.py`).

The two-process launch reproduces `data/accelerate_configs/multi_cpu.yaml` with `torch.distributed.run` rather than
`accelerate launch --config_file`: in this environment (no `mpirun`/`mpiexec`/`mpi4py`), `accelerate launch` cannot
actually start two CPU processes from a `MULTI_CPU` config. Probing it directly confirms this —
`accelerate launch --config_file data/accelerate_configs/multi_cpu.yaml <script>` exits 0 but runs exactly *one*
process (`Accelerator().num_processes == 1`, `distributed_type == DistributedType.NO`) instead of two. The reason is
in `accelerate/commands/launch.py::launch_command`: it special-cases a non-MPI multi-process spawn (via
`torch.distributed.run`) only for `MULTI_GPU`/`FSDP`/`DEEPSPEED`/`MEGATRON_LM`/`XLA` distributed types; `MULTI_CPU`
falls through to `simple_launcher`, which starts a single subprocess unless `--mpirun_hostfile` is given, and that
needs an `mpirun`/`mpiexec` binary that this environment does not have (confirmed via `shutil.which` and
`import mpi4py`). Setting up MPI in a machine shared with concurrent agents was avoided as too invasive for a test
file.

`torch.distributed.run` is what `accelerate launch --multi_gpu` itself calls internally for its own non-MPI
multi-process spawn, so this replicates the same mechanism `accelerate launch` would use, just addressed directly;
`ACCELERATE_USE_CPU`, read from the config's `use_cpu`, is what makes `Accelerator()` resolve to `MULTI_CPU`/gloo
instead of trying (and, on this CPU-only torch build, failing) a GPU backend.
"""

import json
import os
import subprocess
import sys

import pytest
import torch
import yaml

from ..testing_utils import TrlTestCase


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "distillation_multi_teacher_script.py")
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "accelerate_configs", "multi_cpu.yaml")

with open(CONFIG_PATH) as _handle:
    CONFIG = yaml.safe_load(_handle)


@pytest.mark.slow
class TestDistillationTrainerMultiTeacherTwoRankCpu(TrlTestCase):
    def test_two_rank_update_matches_a_single_process_reference(self):
        environment = dict(
            os.environ,
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            TOKENIZERS_PARALLELISM="false",
            ACCELERATE_USE_CPU=str(CONFIG["use_cpu"]),
            ACCELERATE_MIXED_PRECISION=CONFIG["mixed_precision"],
        )

        def run(mode, num_processes):
            output = os.path.join(self.tmp_dir, f"{mode}.json")
            script_args = [
                SCRIPT,
                "--mode",
                mode,
                "--out",
                output,
                "--output-dir",
                os.path.join(self.tmp_dir, f"out-{mode}"),
            ]
            if num_processes > 1:
                # See the module docstring: this is `accelerate launch`'s own non-MPI multi-process mechanism,
                # addressed directly because the CLI does not wire it up for a `MULTI_CPU` config.
                command = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc_per_node={num_processes}",
                    *script_args,
                ]
            else:
                command = [sys.executable, *script_args]
            result = subprocess.run(command, capture_output=True, text=True, timeout=1800, env=environment, cwd=ROOT)
            assert result.returncode == 0, f"{mode} run failed:\n{result.stdout[-4000:]}\n{result.stderr[-8000:]}"
            with open(output) as handle:
                summary = json.load(handle)
            parameters = torch.load(os.path.splitext(output)[0] + "-params.pt", weights_only=True)
            return summary, parameters

        multi, multi_parameters = run("multi", num_processes=CONFIG["num_processes"])
        reference, reference_parameters = run("reference", num_processes=1)

        # The worker asserts this itself; assert it again on the evidence so a single-process fallback can never be
        # read as a two-rank result.
        assert multi["world_size"] == 2
        assert reference["world_size"] == 1

        # Exact: counting quantities the cross-rank reduction must preserve, and both runs trained on the same
        # global batch of tokens (the fixed-completion trainer makes generation deterministic).
        assert multi["optimizer_steps"] == reference["optimizer_steps"] == 2
        assert multi["num_tokens"] == reference["num_tokens"]
        assert multi["teacher_ids"] == reference["teacher_ids"] == ["a", "b"]

        # Both routing IDs were exercised and both report their own per-teacher metrics, in both configurations.
        expected_metric_keys = sorted(
            f"{family}/{teacher_id}" for family in ("teacher_jsd", "teacher_token_frac") for teacher_id in ("a", "b")
        )
        assert multi["teacher_metric_keys"] == expected_metric_keys, multi["teacher_metric_keys"]
        assert reference["teacher_metric_keys"] == expected_metric_keys, reference["teacher_metric_keys"]

        # Within tolerance: the same global update, summed by the cross-rank `accelerator.reduce` of the per-teacher
        # statistics in a different order (two ranks x one microbatch each vs. one process x two rows), and over
        # different microbatch shapes, than the single process does. Floating-point addition is not associative.
        # `atol=1e-6`, `rtol=1e-5` starts from the tolerance used for the equivalent managed-DDP check (`atol=1e-7`,
        # which observed a worst-case absolute difference of 3.7e-9), loosened because this managed path runs an
        # additional per-teacher `accelerator.reduce` that the plain DDP gradient all-reduce does not. The observed
        # worst case here is a loss difference of 6.2e-10 and a parameter difference of 1.5e-8
        # (`model.layers.0.self_attn.k_proj.weight`), both well inside tolerance.
        torch.testing.assert_close(
            torch.tensor(multi["train_losses"]), torch.tensor(reference["train_losses"]), atol=1e-6, rtol=1e-5
        )
        assert sorted(multi_parameters) == sorted(reference_parameters)
        differences = {
            name: (multi_parameters[name] - reference_parameters[name]).abs().max().item()
            for name in reference_parameters
        }
        worst = max(differences, key=differences.get)
        for name, reference_parameter in reference_parameters.items():
            torch.testing.assert_close(
                multi_parameters[name],
                reference_parameter,
                atol=1e-6,
                rtol=1e-5,
                msg=(
                    f"two-rank update differs from the single-process reference: parameter {name} by "
                    f"{differences[name]:.3e} (worst is {worst} by {differences[worst]:.3e})"
                ),
            )
