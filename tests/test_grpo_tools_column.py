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
Tests for the per-sample tool filtering feature in GRPOTrainer.

When a dataset contains a `tools` column, GRPOTrainer automatically uses it to restrict
which tools each sample can call during rollout.
"""

import os
from unittest.mock import patch

import pytest
import torch
import transformers
from datasets import Dataset, load_dataset
from packaging.version import Version

from trl import GRPOConfig, GRPOTrainer

from .testing_utils import TrlTestCase, require_jmespath


# ──────────────────────────────────────────────────────────────────────
# Tool definitions
# ──────────────────────────────────────────────────────────────────────


def multiply_tool(a: int, b: int) -> int:
    """
    Multiplies two integers.

    Args:
        a: The first integer.
        b: The second integer.

    Returns:
        The product of the two integers.
    """
    return a * b


def add_tool(a: int, b: int) -> int:
    """
    Adds two integers.

    Args:
        a: The first integer.
        b: The second integer.

    Returns:
        The sum of the two integers.
    """
    return a + b


async def async_add_tool(a: int, b: int) -> int:
    """
    Asynchronously adds two integers.

    Args:
        a: The first integer.
        b: The second integer.

    Returns:
        The sum of the two integers.
    """
    return a + b


# ──────────────────────────────────────────────────────────────────────
# Unit-level tests (no model loading, fast)
# ──────────────────────────────────────────────────────────────────────


class TestToolsColumnValidation(TrlTestCase):
    """Test that per-sample tool filtering via the `tools` dataset column works correctly."""

    def _make_conversational_dataset(self, tool_names_per_sample):
        """Helper: create a minimal conversational dataset with a tools column."""
        prompts = [[{"role": "user", "content": f"Question {i}"}] for i in range(len(tool_names_per_sample))]
        return Dataset.from_dict({"prompt": prompts, "tools": tool_names_per_sample})

    @pytest.mark.xfail(
        condition=Version(transformers.__version__) < Version("5.0.0"),
        reason="Tool parsing is not supported in transformers versions below 5.0.0",
        strict=True,
    )
    @require_jmespath
    def test_valid_tools_column_passes_validation(self):
        """A dataset with a tools column and matching tool pool should init without errors."""
        dataset = self._make_conversational_dataset([["multiply_tool"], ["add_tool"], ["multiply_tool", "add_tool"]])
        trainer = GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen3MoeForCausalLM",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=GRPOConfig(
                output_dir=self.tmp_dir,
                report_to="none",
                per_device_train_batch_size=3,
                num_generations=3,
            ),
            train_dataset=dataset,
            tools=[multiply_tool, add_tool],
        )
        assert {t.__name__ for t in trainer.tools} == {"multiply_tool", "add_tool"}

    @pytest.mark.xfail(
        condition=Version(transformers.__version__) < Version("5.0.0"),
        reason="Tool parsing is not supported in transformers versions below 5.0.0",
        strict=True,
    )
    @require_jmespath
    def test_no_tools_column_backward_compat(self):
        """When the dataset has no tools column, trainer behaves as before (all tools used)."""
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        trainer = GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen3MoeForCausalLM",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=GRPOConfig(
                output_dir=self.tmp_dir,
                report_to="none",
                per_device_train_batch_size=3,
                num_generations=3,
            ),
            train_dataset=dataset,
            tools=[multiply_tool],
        )
        assert {t.__name__ for t in trainer.tools} == {"multiply_tool"}

    @pytest.mark.xfail(
        condition=Version(transformers.__version__) < Version("5.0.0"),
        reason="Tool parsing is not supported in transformers versions below 5.0.0",
        strict=True,
    )
    @require_jmespath
    def test_tools_accessible_per_sample(self):
        """Each per-sample tool dict should contain the correct callables for its sample."""
        dataset = self._make_conversational_dataset([["multiply_tool", "add_tool"]])
        trainer = GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen3MoeForCausalLM",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=GRPOConfig(
                output_dir=self.tmp_dir,
                report_to="none",
                per_device_train_batch_size=3,
                num_generations=3,
            ),
            train_dataset=dataset,
            tools=[multiply_tool, add_tool],
        )
        assert trainer._sync_tool_dicts[0]["multiply_tool"] is multiply_tool
        assert trainer._sync_tool_dicts[0]["add_tool"] is add_tool

    @pytest.mark.xfail(
        condition=Version(transformers.__version__) < Version("5.0.0"),
        reason="Tool parsing is not supported in transformers versions below 5.0.0",
        strict=True,
    )
    @require_jmespath
    def test_signature_columns_include_tools(self):
        """The `tools` column should always be included in _signature_columns."""
        dataset = self._make_conversational_dataset([["multiply_tool"]])
        trainer = GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen3MoeForCausalLM",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=GRPOConfig(
                output_dir=self.tmp_dir,
                report_to="none",
                per_device_train_batch_size=3,
                num_generations=3,
            ),
            train_dataset=dataset,
            tools=[multiply_tool],
        )
        trainer._set_signature_columns_if_needed()
        assert "tools" in trainer._signature_columns


# ──────────────────────────────────────────────────────────────────────
# Integration tests (model loading + training with fake_generate)
# ──────────────────────────────────────────────────────────────────────


class TestToolsColumnTraining(TrlTestCase):
    """End-to-end tests for training with per-sample tool filtering."""

    @pytest.mark.xfail(
        condition=Version(transformers.__version__) < Version("5.0.0"),
        reason="Tool parsing is not supported in transformers versions below 5.0.0",
        strict=True,
    )
    @require_jmespath
    def test_training_with_tools_column(self):
        """Train with a `tools` dataset column and verify per-sample filtering works end-to-end.

        We create a 3-sample dataset where:
          - Sample 0: only multiply_tool available  → model calls multiply_tool (succeeds)
          - Sample 1: only multiply_tool available  → model calls multiply_tool (fails, wrong arg)
          - Sample 2: only multiply_tool available  → model returns plain text (no tool call)
        """
        dataset = Dataset.from_dict(
            {
                "prompt": [
                    [{"role": "user", "content": "What is 3 times 4?"}],
                    [{"role": "user", "content": "Multiply 5 and 6."}],
                    [{"role": "user", "content": "Tell me a joke."}],
                ],
                "tools": [
                    ["multiply_tool"],
                    ["multiply_tool"],
                    ["multiply_tool"],
                ],
            }
        )

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=128,
            report_to="none",
        )
        trainer = GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen3MoeForCausalLM",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
            tools=[multiply_tool, add_tool],  # global pool has both; dataset column restricts per sample
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        def fake_generate(input_ids, **kwargs):
            if input_ids.shape[0] == 3:  # first call
                # fmt: off
                completion_ids = torch.tensor(
                    [
                        # '<tool_call>\n{"name": "multiply_tool", "arguments": {"a": 3, "b": 4}}\n</tool_call><|im_end|>'
                        [151657, 198, 4913, 606, 788, 330, 64648, 22785, 497, 330, 16370, 788, 5212, 64, 788, 220, 18, 11, 330, 65, 788, 220, 19, 11248, 151658, 151645],
                        # invalid tool call: wrong argument name "c" instead of "b"
                        # '<tool_call>\n{"name": "multiply_tool", "arguments": {"a": 3, "c": 4}}\n</tool_call><|im_end|>'
                        [151657, 198, 4913, 606, 788, 330, 64648, 22785, 497, 330, 16370, 788, 5212, 64, 788, 220, 18, 11, 330, 66, 788, 220, 19, 11248, 151658, 151645],
                        # "I don't know any tool<|im_end|>"
                        [40, 1513, 944, 1414, 894, 5392, 151645, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643],
                    ],
                    device=input_ids.device,
                )
                # fmt: on
            else:  # second call: only 2 samples had tool calls
                completion_ids = torch.tensor(
                    [
                        # 'Done!<|im_end|>'
                        [17453, 0, 151645],
                        # 'Done!<|im_end|>'
                        [17453, 0, 151645],
                    ],
                    device=input_ids.device,
                )
            return torch.cat([input_ids, completion_ids], dim=-1)

        with patch.object(trainer.model, "generate", side_effect=fake_generate):
            trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        assert trainer.state.log_history[-1]["tools/call_frequency"] == pytest.approx(2 / 3)
        assert trainer.state.log_history[-1]["tools/failure_frequency"] == pytest.approx(1 / 2)

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.xfail(
        condition=Version(transformers.__version__) < Version("5.0.0"),
        reason="Tool parsing is not supported in transformers versions below 5.0.0",
        strict=True,
    )
    @require_jmespath
    def test_training_with_tools_column_subset(self):
        """Verify that the model only sees the subset of tools declared per sample.

        We have two tools (multiply_tool, add_tool) but each sample only allows one.
        Because the model's fake output calls multiply_tool, the sample that only allows add_tool
        should fail the tool call (tool not found) — proving the filtering works.
        """
        dataset = Dataset.from_dict(
            {
                "prompt": [
                    [{"role": "user", "content": "What is 3 times 4?"}],
                    [{"role": "user", "content": "Add 1 and 2."}],
                    [{"role": "user", "content": "Tell me a joke."}],
                ],
                "tools": [
                    ["multiply_tool"],  # sample 0: only multiply allowed
                    ["add_tool"],  # sample 1: only add allowed → multiply_tool call will fail ("not found")
                    ["multiply_tool", "add_tool"],  # sample 2: both allowed, but model won't call any
                ],
            }
        )

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=128,
            report_to="none",
        )
        trainer = GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen3MoeForCausalLM",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
            tools=[multiply_tool, add_tool],
        )

        def fake_generate(input_ids, **kwargs):
            if input_ids.shape[0] == 3:  # first call
                # fmt: off
                completion_ids = torch.tensor(
                    [
                        # Sample 0: calls multiply_tool → should succeed (multiply_tool is allowed)
                        # '<tool_call>\n{"name": "multiply_tool", "arguments": {"a": 3, "b": 4}}\n</tool_call><|im_end|>'
                        [151657, 198, 4913, 606, 788, 330, 64648, 22785, 497, 330, 16370, 788, 5212, 64, 788, 220, 18, 11, 330, 65, 788, 220, 19, 11248, 151658, 151645],
                        # Sample 1: calls multiply_tool → should FAIL (only add_tool is allowed)
                        # '<tool_call>\n{"name": "multiply_tool", "arguments": {"a": 3, "b": 4}}\n</tool_call><|im_end|>'
                        [151657, 198, 4913, 606, 788, 330, 64648, 22785, 497, 330, 16370, 788, 5212, 64, 788, 220, 18, 11, 330, 65, 788, 220, 19, 11248, 151658, 151645],
                        # Sample 2: plain text, no tool call
                        # "I don't know any tool<|im_end|>"
                        [40, 1513, 944, 1414, 894, 5392, 151645, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643],
                    ],
                    device=input_ids.device,
                )
                # fmt: on
            else:  # second call: 2 samples had tool calls
                completion_ids = torch.tensor(
                    [
                        [17453, 0, 151645],  # 'Done!<|im_end|>'
                        [17453, 0, 151645],  # 'Done!<|im_end|>'
                    ],
                    device=input_ids.device,
                )
            return torch.cat([input_ids, completion_ids], dim=-1)

        with patch.object(trainer.model, "generate", side_effect=fake_generate):
            trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        # 2 out of 3 samples per batch made tool calls
        assert trainer.state.log_history[-1]["tools/call_frequency"] == pytest.approx(2 / 3)
        # At least some calls should fail because sample 1 only allows add_tool but model calls multiply_tool.
        # The exact failure rate depends on batch composition across epochs/steps.
        assert trainer.state.log_history[-1]["tools/failure_frequency"] > 0

    @pytest.mark.xfail(
        condition=Version(transformers.__version__) < Version("5.0.0"),
        reason="Tool parsing is not supported in transformers versions below 5.0.0",
        strict=True,
    )
    @require_jmespath
    def test_training_without_tools_column_backward_compat(self):
        """Training with tools but without a `tools` dataset column should work exactly as before."""
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=128,
            report_to="none",
        )
        trainer = GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen3MoeForCausalLM",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
            tools=[multiply_tool],
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        def fake_generate(input_ids, **kwargs):
            if input_ids.shape[0] == 3:
                # fmt: off
                completion_ids = torch.tensor(
                    [
                        # '<tool_call>\n{"name": "multiply_tool", "arguments": {"a": 3, "b": 4}}\n</tool_call><|im_end|>'
                        [151657, 198, 4913, 606, 788, 330, 64648, 22785, 497, 330, 16370, 788, 5212, 64, 788, 220, 18, 11, 330, 65, 788, 220, 19, 11248, 151658, 151645],
                        # invalid tool call: wrong argument name
                        [151657, 198, 4913, 606, 788, 330, 64648, 22785, 497, 330, 16370, 788, 5212, 64, 788, 220, 18, 11, 330, 66, 788, 220, 19, 11248, 151658, 151645],
                        # "I don't know any tool<|im_end|>"
                        [40, 1513, 944, 1414, 894, 5392, 151645, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643],
                    ],
                    device=input_ids.device,
                )
                # fmt: on
            else:
                completion_ids = torch.tensor(
                    [[17453, 0, 151645], [17453, 0, 151645]],
                    device=input_ids.device,
                )
            return torch.cat([input_ids, completion_ids], dim=-1)

        with patch.object(trainer.model, "generate", side_effect=fake_generate):
            trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        assert trainer.state.log_history[-1]["tools/call_frequency"] == pytest.approx(2 / 3)
        assert trainer.state.log_history[-1]["tools/failure_frequency"] == pytest.approx(1 / 2)

        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.xfail(
        condition=Version(transformers.__version__) < Version("5.0.0"),
        reason="Tool parsing is not supported in transformers versions below 5.0.0",
        strict=True,
    )
    @require_jmespath
    def test_training_with_none_in_tools_column_falls_back(self):
        """When a sample's tools column is None, fall back to the full global tools list."""
        dataset = Dataset.from_dict(
            {
                "prompt": [
                    [{"role": "user", "content": "What is 3 times 4?"}],
                    [{"role": "user", "content": "What is 5 plus 6?"}],
                    [{"role": "user", "content": "Tell me a joke."}],
                ],
                "tools": [
                    ["multiply_tool"],  # only multiply
                    None,  # fallback to all tools (multiply + add)
                    ["multiply_tool", "add_tool"],  # both tools (fake_generate calls multiply_tool for all batches)
                ],
            }
        )

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=128,
            report_to="none",
        )
        trainer = GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen3MoeForCausalLM",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
            tools=[multiply_tool, add_tool],
        )

        def fake_generate(input_ids, **kwargs):
            if input_ids.shape[0] == 3:
                # fmt: off
                completion_ids = torch.tensor(
                    [
                        # Sample 0: calls multiply_tool → ok (allowed)
                        [151657, 198, 4913, 606, 788, 330, 64648, 22785, 497, 330, 16370, 788, 5212, 64, 788, 220, 18, 11, 330, 65, 788, 220, 19, 11248, 151658, 151645],
                        # Sample 1: calls multiply_tool → ok (None fallback = all tools)
                        [151657, 198, 4913, 606, 788, 330, 64648, 22785, 497, 330, 16370, 788, 5212, 64, 788, 220, 18, 11, 330, 65, 788, 220, 19, 11248, 151658, 151645],
                        # Sample 2: plain text
                        [40, 1513, 944, 1414, 894, 5392, 151645, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643],
                    ],
                    device=input_ids.device,
                )
                # fmt: on
            else:
                completion_ids = torch.tensor(
                    [[17453, 0, 151645], [17453, 0, 151645]],
                    device=input_ids.device,
                )
            return torch.cat([input_ids, completion_ids], dim=-1)

        with patch.object(trainer.model, "generate", side_effect=fake_generate):
            trainer.train()

        # Both tool calls should succeed (sample 0 has multiply allowed, sample 1 has all tools via None fallback)
        assert trainer.state.log_history[-1]["train_loss"] is not None
        assert trainer.state.log_history[-1]["tools/call_frequency"] == pytest.approx(2 / 3)
        # No failures: both samples' tool calls are to multiply_tool which is in their allowed set
        assert trainer.state.log_history[-1]["tools/failure_frequency"] == pytest.approx(0.0)

    @pytest.mark.xfail(
        condition=Version(transformers.__version__) < Version("5.0.0"),
        reason="Tool parsing is not supported in transformers versions below 5.0.0",
        strict=True,
    )
    @require_jmespath
    def test_training_with_async_tool_and_tools_column(self):
        """Verify that async tools also work with per-sample filtering."""
        dataset = Dataset.from_dict(
            {
                "prompt": [
                    [{"role": "user", "content": "What is 3 times 4?"}],
                    [{"role": "user", "content": "What is 5 plus 6?"}],
                    [{"role": "user", "content": "Tell me a joke."}],
                ],
                "tools": [
                    ["multiply_tool"],
                    ["multiply_tool"],
                    ["multiply_tool"],
                ],
            }
        )

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=128,
            report_to="none",
        )
        trainer = GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen3MoeForCausalLM",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
            tools=[multiply_tool, async_add_tool],  # mix of sync and async in global pool
        )

        def fake_generate(input_ids, **kwargs):
            if input_ids.shape[0] == 3:
                # fmt: off
                completion_ids = torch.tensor(
                    [
                        [151657, 198, 4913, 606, 788, 330, 64648, 22785, 497, 330, 16370, 788, 5212, 64, 788, 220, 18, 11, 330, 65, 788, 220, 19, 11248, 151658, 151645],
                        [151657, 198, 4913, 606, 788, 330, 64648, 22785, 497, 330, 16370, 788, 5212, 64, 788, 220, 18, 11, 330, 66, 788, 220, 19, 11248, 151658, 151645],
                        [40, 1513, 944, 1414, 894, 5392, 151645, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643],
                    ],
                    device=input_ids.device,
                )
                # fmt: on
            else:
                completion_ids = torch.tensor(
                    [[17453, 0, 151645], [17453, 0, 151645]],
                    device=input_ids.device,
                )
            return torch.cat([input_ids, completion_ids], dim=-1)

        with patch.object(trainer.model, "generate", side_effect=fake_generate):
            trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

    @pytest.mark.xfail(
        condition=Version(transformers.__version__) < Version("5.2.0"),
        reason="Environment factory support is not available in transformers versions below 5.2.0",
        strict=True,
    )
    @require_jmespath
    @patch.dict(os.environ, {"TRL_EXPERIMENTAL_SILENCE": "1"})
    def test_training_with_tools_column_and_environment_factory(self):
        """Verify per-sample tool filtering uses correctly-bound per-environment methods.

        Each sample i should call _sync_tool_dicts[i]["increment"] (bound to environments[i]).
        If the old _tool_registry fallback were used, all samples would resolve to
        environments[0].increment, and only environments[0]._counter would be updated.
        """
        dataset = Dataset.from_dict(
            {
                "prompt": [
                    [{"role": "user", "content": "Increment by 1."}],
                    [{"role": "user", "content": "Increment by 1."}],
                    [{"role": "user", "content": "Tell me a joke."}],
                ],
                "tools": [
                    ["increment"],
                    ["increment"],
                    ["increment"],
                ],
            }
        )

        class DummyEnvironment:
            def reset(self, **kwargs):
                self._counter = 0

            def increment(self, step: int) -> int:
                """
                Increment the internal counter.

                Args:
                    step: Value to add to the counter.

                Returns:
                    The updated counter value.
                """
                self._counter += step
                return self._counter

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=128,
            report_to="none",
        )

        trainer = GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen3MoeForCausalLM",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
            environment_factory=DummyEnvironment,
        )

        def fake_generate(input_ids, **kwargs):
            if input_ids.shape[0] == 3:  # first call
                # fmt: off
                completion_ids = torch.tensor(
                    [
                        # Sample 0: '<tool_call>\n{"name": "increment", "arguments": {"step": 1}}\n</tool_call><|im_end|>'
                        [151657, 198, 4913, 606, 788, 330, 35744, 497, 330, 16370, 788, 5212, 9520, 788, 220, 16, 11248, 151658, 151645, 151643],
                        # Sample 1: '<tool_call>\n{"name": "increment", "arguments": {"step": 1}}\n</tool_call><|im_end|>'
                        [151657, 198, 4913, 606, 788, 330, 35744, 497, 330, 16370, 788, 5212, 9520, 788, 220, 16, 11248, 151658, 151645, 151643],
                        # Sample 2: "I won't increment<|im_end|>"
                        [40, 2765, 944, 16252, 151645, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643],
                    ],
                    device=input_ids.device,
                )
                # fmt: on
            else:  # second call: 2 samples had tool calls
                completion_ids = torch.tensor(
                    [
                        [17453, 0, 151645],  # 'Done!<|im_end|>'
                        [17453, 0, 151645],  # 'Done!<|im_end|>'
                    ],
                    device=input_ids.device,
                )
            return torch.cat([input_ids, completion_ids], dim=-1)

        with patch.object(trainer.model, "generate", side_effect=fake_generate):
            trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        assert trainer.state.log_history[-1]["tools/call_frequency"] == pytest.approx(2 / 3)
        assert trainer.state.log_history[-1]["tools/failure_frequency"] == pytest.approx(0.0)
        # Verify correct per-environment binding: each environment's own increment was called.
        # If all calls resolved to environments[0] (old registry fallback), then
        # environments[0]._counter would be 2 and environments[1]._counter would be 0.
        assert trainer.environments[0]._counter == 1
        assert trainer.environments[1]._counter == 1
        assert trainer.environments[2]._counter == 0
