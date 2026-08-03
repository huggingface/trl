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

import json
import os
import subprocess
from pathlib import Path

import pytest
import torch
import transformers
from packaging.version import Version

from ..testing_utils import TrlTestCase, require_liger_kernel, require_torch_multi_accelerator


ROOT = Path(__file__).resolve().parents[2]


def run_command(command: list[str], env: dict[str, str]) -> None:
    result = subprocess.run(command, env=env, cwd=ROOT)
    assert result.returncode == 0


@pytest.fixture
def get_config_path(lazy_shared_datadir):
    def _get_config_path(config_name):
        return lazy_shared_datadir / "accelerate_configs" / f"{config_name}.yaml"

    return _get_config_path


@require_torch_multi_accelerator
class TestDistributed(TrlTestCase):
    @pytest.mark.parametrize(
        "config",
        [
            "ddp",
            pytest.param(
                "zero2",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
            pytest.param(
                "zero3",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
            "fsdp2",
        ],
    )
    def test_sft(self, config, get_config_path):
        # fmt: off
        run_command(
            [
                "accelerate", "launch", "--config_file", get_config_path(config), "trl/scripts/sft.py",
                "--output_dir", self.tmp_dir,
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name", "trl-internal-testing/zen",
                "--dataset_config", "standard_language_modeling",
            ],
            os.environ.copy(),
        )
        # fmt: on

    @pytest.mark.parametrize(
        "config",
        [
            "ddp",
            pytest.param(
                "zero2",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
            pytest.param(
                "zero3",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
            "fsdp2",
        ],
    )
    def test_sft_nll_loss(self, config, get_config_path):
        # fmt: off
        run_command(
            [
                "accelerate", "launch", "--config_file", get_config_path(config), "trl/scripts/sft.py",
                "--output_dir", self.tmp_dir,
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name", "trl-internal-testing/zen",
                "--dataset_config", "standard_language_modeling",
                "--loss_type", "nll",
            ],
            os.environ.copy(),
        )
        # fmt: on

    @pytest.mark.parametrize(
        "config",
        [
            "ddp",
            pytest.param(
                "zero2",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
            pytest.param(
                "zero3",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
            "fsdp2",
        ],
    )
    def test_dpo(self, config, get_config_path):
        # fmt: off
        run_command(
            [
                "accelerate", "launch", "--config_file", get_config_path(config), "trl/scripts/dpo.py",
                "--output_dir", self.tmp_dir,
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name", "trl-internal-testing/zen",
                "--dataset_config", "standard_preference",
            ],
            os.environ.copy(),
        )
        # fmt: on

    @pytest.mark.parametrize(
        "config",
        [
            "ddp",
            pytest.param(
                "zero2",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
            pytest.param(
                "zero3",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
        ],
    )
    def test_dpo_precompute_ref_log_probs(self, config, get_config_path):
        # `--eval_strategy epoch` passes an eval dataset, so reference log-probs are precomputed for both the train and
        # eval splits (two passes), which is what previously broke multi-GPU precompute (fingerprint cache mismatch, and
        # a corrupted ZeRO-3 parameter coordinator from re-initializing DeepSpeed on the policy model per pass).
        # fmt: off
        run_command(
            [
                "accelerate", "launch", "--config_file", get_config_path(config), "trl/scripts/dpo.py",
                "--output_dir", self.tmp_dir,
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name", "trl-internal-testing/zen",
                "--dataset_config", "standard_preference",
                "--precompute_ref_log_probs",
                "--eval_strategy", "epoch",
            ],
            os.environ.copy(),
        )
        # fmt: on

    @require_liger_kernel
    @pytest.mark.parametrize(
        "config",
        [
            "ddp",
            pytest.param(
                "zero2",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
            pytest.param(
                "zero3",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
            pytest.param(
                "fsdp2",
                marks=pytest.mark.xfail(
                    reason="Liger DPO loss reads `lm_head.weight` and runs the backbone directly, which is "
                    "incompatible with FSDP2's DTensor-sharded parameters (mixed Tensor/DTensor ops).",
                    strict=True,
                ),
            ),
        ],
    )
    def test_dpo_liger(self, config, get_config_path):
        # fmt: off
        run_command(
            [
                "accelerate", "launch", "--config_file", get_config_path(config), "trl/scripts/dpo.py",
                "--output_dir", self.tmp_dir,
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name", "trl-internal-testing/zen",
                "--dataset_config", "standard_preference",
                "--use_liger_kernel",
            ],
            os.environ.copy(),
        )
        # fmt: on

    @require_liger_kernel
    @pytest.mark.parametrize(
        "config",
        [
            "ddp",
            pytest.param(
                "zero2",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
            pytest.param(
                "zero3",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
            pytest.param(
                "fsdp2",
                marks=pytest.mark.xfail(
                    reason="Liger KTO loss reads `lm_head.weight` and runs the backbone directly, which is "
                    "incompatible with FSDP2's DTensor-sharded parameters (mixed Tensor/DTensor ops).",
                    strict=True,
                ),
            ),
        ],
    )
    def test_kto_liger(self, config, get_config_path):
        # fmt: off
        run_command(
            [
                "accelerate", "launch", "--config_file", get_config_path(config), "trl/scripts/kto.py",
                "--output_dir", self.tmp_dir,
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name", "trl-internal-testing/zen",
                "--dataset_config", "standard_unpaired_preference",
                "--use_liger_kernel",
            ],
            os.environ.copy(),
        )
        # fmt: on

    @pytest.mark.parametrize(
        "config",
        [
            "ddp",
            pytest.param(
                "zero2",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
            pytest.param(
                "zero3",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
            "fsdp2",
        ],
    )
    def test_sft_dataset_streaming(self, config, get_config_path):
        # fmt: off
        run_command(
            [
                "accelerate", "launch", "--config_file", get_config_path(config), "trl/scripts/sft.py",
                "--output_dir", self.tmp_dir,
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name", "trl-internal-testing/zen",
                "--dataset_config", "standard_language_modeling",
                "--dataset_streaming",
                "--max_steps", "3",
            ],
            os.environ.copy(),
        )
        # fmt: on

    @pytest.mark.parametrize(
        "config",
        [
            "ddp",
            pytest.param(
                "zero2",
                marks=pytest.mark.xfail(
                    condition=Version("2.10") <= Version(torch.__version__)
                    and Version(transformers.__version__) < Version("5.1.0"),
                    reason="ZeRO 2 + PEFT was failing before transformers 5.1.0 on torch 2.10; see #4884",
                ),
            ),
            pytest.param(
                "zero3",
                marks=pytest.mark.xfail(
                    condition=Version("2.10") <= Version(torch.__version__)
                    and Version(transformers.__version__) < Version("5.1.0"),
                    reason="ZeRO 3 + PEFT was failing before transformers 5.1.0 on torch 2.10; see #4884",
                ),
            ),
            "fsdp2",
        ],
    )
    def test_sft_peft(self, config, get_config_path):
        # fmt: off
        run_command(
            [
                "accelerate", "launch", "--config_file", get_config_path(config), "trl/scripts/sft.py",
                "--output_dir", self.tmp_dir,
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name", "trl-internal-testing/zen",
                "--dataset_config", "standard_language_modeling",
                "--use_peft",
            ],
            os.environ.copy(),
        )
        # fmt: on

    @pytest.mark.parametrize(
        "config",
        [
            "ddp",
            pytest.param(
                "zero2",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
            pytest.param(
                "zero3",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
            "fsdp2",
        ],
    )
    def test_reward(self, config, get_config_path):
        # fmt: off
        run_command(
            [
                "accelerate", "launch", "--config_file", get_config_path(config), "trl/scripts/reward.py",
                "--output_dir", self.tmp_dir,
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name", "trl-internal-testing/zen",
                "--dataset_config", "conversational_implicit_prompt_preference",
            ],
            os.environ.copy(),
        )
        # fmt: on

    @pytest.mark.parametrize(
        "config",
        [
            "ddp",
            pytest.param(
                "zero2",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
            pytest.param(
                "zero3",
                marks=pytest.mark.xfail(
                    Version("5.0.0") <= Version(transformers.__version__) < Version("5.5.4"),
                    reason="ZeRO-3 fails with transformers >= 5.0.0 and < 5.5.4 (fixed in transformers#45414), see #4899",
                    strict=True,
                ),
            ),
            pytest.param(
                "fsdp2",
                marks=pytest.mark.skipif(
                    Version("5.4.0") <= Version(transformers.__version__) < Version("5.6.0"),
                    reason="Upstream issue: NaN weights on non-rank-0 FSDP processes (see #5386 and transformers#45050)",
                ),
            ),
        ],
    )
    def test_rloo(self, config, get_config_path):
        # fmt: off
        run_command(
            [
                "accelerate", "launch", "--config_file", get_config_path(config), "trl/scripts/rloo.py",
                "--output_dir", self.tmp_dir,
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name", "trl-internal-testing/zen",
                "--dataset_config", "conversational_prompt_only",
                "--reward_model_name_or_path", "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            ],
            os.environ.copy(),
        )
        # fmt: on

    @pytest.mark.parametrize(
        "config",
        [
            "ddp",
            pytest.param(
                "zero2",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
            pytest.param(
                "zero3",
                marks=pytest.mark.xfail(
                    Version("5.0.0") <= Version(transformers.__version__) < Version("5.5.4"),
                    reason="ZeRO-3 fails with transformers >= 5.0.0 and < 5.5.4 (fixed in transformers#45414), see #4899",
                    strict=True,
                ),
            ),
            "fsdp2",
        ],
    )
    def test_grpo(self, config, get_config_path):
        # fmt: off
        run_command(
            [
                "accelerate", "launch", "--config_file", get_config_path(config), "trl/scripts/grpo.py",
                "--output_dir", self.tmp_dir,
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name", "trl-internal-testing/zen",
                "--dataset_config", "conversational_prompt_only",
                "--reward_model_name_or_path", "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            ],
            os.environ.copy(),
        )
        # fmt: on

    @require_liger_kernel
    @pytest.mark.parametrize(
        "config",
        [
            "ddp",
            pytest.param(
                "zero2",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) == Version("5.1.0"),
                    reason="Upstream incompatibility: deepspeed and transformers==5.1.0 (see transformers#43780)",
                ),
            ),
            pytest.param(
                "zero3",
                marks=pytest.mark.xfail(
                    Version("5.0.0") <= Version(transformers.__version__) < Version("5.5.4"),
                    reason="ZeRO-3 fails with transformers >= 5.0.0 and < 5.5.4 (fixed in transformers#45414), see #4899",
                    strict=True,
                ),
            ),
            "fsdp2",
        ],
    )
    def test_grpo_liger(self, config, get_config_path):
        # fmt: off
        run_command(
            [
                "accelerate", "launch", "--config_file", get_config_path(config), "trl/scripts/grpo.py",
                "--output_dir", self.tmp_dir,
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name", "trl-internal-testing/zen",
                "--dataset_config", "conversational_prompt_only",
                "--reward_model_name_or_path", "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                "--use_liger_kernel",
            ],
            os.environ.copy(),
        )
        # fmt: on

    def test_sft_chunked_nll_fsdp2_no_per_chunk_allgather(self, lazy_shared_datadir):
        # Perf-regression guard for the PR #6077 class: a chunked cross-entropy path must NOT re-gather the
        # sharded `lm_head.weight` once per token chunk under FSDP2 (correct loss, silently slow, invisible
        # to a pass/fail test). The companion worker runs one SFT `chunked_nll` step under a 2-process FSDP2
        # group (reshard_after_forward=True — the condition that triggers the bug) and counts the all-gather
        # collectives during that step via CommDebugMode (torch's DTensor-native comm counter; required
        # because under FSDP2 the parameter unshard is driven by autograd hooks / c10d collectives, not by
        # `DTensor.full_tensor()`).
        #
        # `_chunked_cross_entropy_loss` chunks over VALID TOKENS, not vocab, so the measured count mixes two
        # components: a chunk-INDEPENDENT FSDP2 unshard baseline B (one gather per sharded param per fwd/bwd)
        # and, only if the bug is present, ~ceil(n_valid / chunk_size) per-chunk lm_head re-gathers. To
        # separate them soundly, the worker is run TWICE: a "baseline" run with a chunk size large enough for a
        # single token chunk (regression signal = 0, so it measures B directly), then a "probe" run with a
        # shrunk chunk size (many token chunks). The regression assertion bounds the probe's EXCESS over the
        # measured baseline, so it stays non-vacuous even when B is large relative to the token-chunk count.
        worker = Path(__file__).parent / "_chunked_nll_allgather_worker.py"
        config_path = lazy_shared_datadir / "accelerate_configs" / "fsdp2_reshard.yaml"
        prefix = "CHUNKED_NLL_ALLGATHER_RESULT"

        def _run(mode: str) -> dict:
            # Pin the repo root onto PYTHONPATH for the child: `accelerate launch` re-execs each rank via
            # torch.distributed.elastic, which sets sys.path[0] to the launched script's directory, not cwd.
            # Without this, a non-editable `trl` already in site-packages would shadow the working tree.
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
            env["CHUNKED_NLL_MODE"] = mode
            result = subprocess.run(
                ["accelerate", "launch", "--config_file", str(config_path), str(worker)],
                env=env,
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, (
                f"worker ({mode}) failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
            lines = [ln for ln in result.stdout.splitlines() if ln.startswith(prefix)]
            assert len(lines) == 1, (
                f"expected exactly one result line from the {mode} run, got {lines}\n{result.stdout}"
            )
            return json.loads(lines[0][len(prefix) :].strip())

        # Baseline run: one token chunk => regression signal = 0, so `all_gathers` is the pure FSDP2 unshard
        # baseline B. B is chunk-independent (set by the model's parameter/sharding structure), so it is also
        # the count the fixed probe run should land on. Measuring it — rather than assuming it — is what makes
        # the regression bound sound even when B is large relative to the token-chunk count.
        baseline = _run("baseline")
        assert baseline["loss_finite"], f"baseline chunked_nll loss not finite under FSDP2: {baseline}"
        assert baseline["n_chunks_if_regressed"] == 1, (
            f"baseline run was expected to execute exactly one token chunk (regression signal = 0): {baseline}"
        )
        baseline_gathers = baseline["all_gathers"]

        # Probe run: shrink the chunk size so many token chunks run. A per-chunk `lm_head.weight` re-gather
        # regression would then add ~n_chunks all-gathers on top of the baseline.
        probe = _run("probe")
        assert probe["loss_finite"], f"chunked_nll loss not finite under FSDP2: {probe}"
        observed = probe["all_gathers"]
        n_chunks = probe["n_chunks_if_regressed"]

        # Non-vacuity guard: a per-token-chunk regression can only be detected if the step actually ran several
        # token chunks. If only one chunk ran, a regression would gather exactly once too and the test would
        # pass for the wrong reason. Require a comfortably multi-chunk run.
        assert n_chunks > 4, (
            f"test is vacuous — only {n_chunks} token chunk(s) ran in the probe, so a per-chunk regression "
            f"could not be observed; increase batch/length or shrink chunk_size: {probe}"
        )
        # Regression check. A per-chunk regather would do ~n_chunks all-gathers of lm_head.weight ON TOP of the
        # baseline B; the fixed path stays at ~B regardless of the chunk count. Comparing the probe's count
        # against the MEASURED baseline (not against n_chunks directly) removes the chunk-independent FSDP2
        # unshards from the comparison — which is exactly what makes the bound non-vacuous when B is large. The
        # excess over baseline must stay far below the regression magnitude; bound it by n_chunks // 4 so the
        # ceiling tracks the token/chunk arithmetic rather than any hardcoded collective count.
        excess = observed - baseline_gathers
        assert excess <= n_chunks // 4, (
            f"per-chunk lm_head.weight all-gathers detected (#6077 regression): probe did {observed} all-gathers "
            f"vs baseline {baseline_gathers} (excess {excess}) over {n_chunks} token chunks: {probe}"
        )
