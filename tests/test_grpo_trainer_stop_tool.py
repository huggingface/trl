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

"""Tests for GRPOConfig.stop_tool_names — stop-tool termination in the agent loop.

These tests verify that when a tool name is listed in ``stop_tool_names``:

1. The ``GRPOConfig`` field is stored correctly.
2. ``model.generate`` is NOT called again after the designated stop tool fires
   (i.e. no spurious post-tool generation turn).
3. When only *some* samples call the stop tool, generation continues only for
   the remaining samples.
4. ``completion_ids`` include the tool-result tokens so the reward function
   sees the full conversation.
5. The feature is a strict no-op when ``stop_tool_names=None`` (existing
   behaviour is preserved).
"""

import gc
from unittest.mock import patch

import pytest
import torch
import transformers
from datasets import load_dataset
from packaging.version import Version
from transformers import AutoTokenizer

from trl import GRPOConfig, GRPOTrainer

from .testing_utils import TrlTestCase, require_jmespath


# ---------------------------------------------------------------------------
# Minimal tools
# ---------------------------------------------------------------------------


def multiply_tool(a: int, b: int) -> int:
    """Multiply two integers together.

    Args:
        a: First integer.
        b: Second integer.

    Returns:
        The product of a and b.
    """
    return a * b


def final_answer(answer: int) -> int:
    """Submit the final answer and terminate agent execution.

    Args:
        answer: The final integer answer to submit.

    Returns:
        The submitted answer (echoed back).
    """
    return answer


# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

_MODEL_ID = "trl-internal-testing/tiny-Qwen3MoeForCausalLM"
_REWARD_MODEL_ID = "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5"
_DATASET = "trl-internal-testing/zen"


def _make_final_answer_ids(tokenizer) -> list[int]:
    """Token IDs for ``<tool_call>\\n{"name": "final_answer", ...}\\n</tool_call><|im_end|>``."""
    text = '<tool_call>\n{"name": "final_answer", "arguments": {"answer": 7}}\n</tool_call>'
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids + [tokenizer.convert_tokens_to_ids("<|im_end|>")]


def _make_multiply_ids(tokenizer) -> list[int]:
    """Token IDs for a valid ``multiply_tool`` call completion."""
    text = '<tool_call>\n{"name": "multiply_tool", "arguments": {"a": 3, "b": 4}}\n</tool_call>'
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids + [tokenizer.convert_tokens_to_ids("<|im_end|>")]


def _make_done_ids(tokenizer) -> list[int]:
    """Token IDs for a plain ``Done!<|im_end|>`` text completion (no tool call)."""
    ids = tokenizer.encode("Done!", add_special_tokens=False)
    return ids + [tokenizer.convert_tokens_to_ids("<|im_end|>")]


def _base_config(tmp_dir, **kwargs) -> GRPOConfig:
    return GRPOConfig(
        output_dir=tmp_dir,
        learning_rate=0.1,
        per_device_train_batch_size=3,
        num_generations=3,
        max_completion_length=128,
        report_to="none",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Config-level tests (no GPU / model required)
# ---------------------------------------------------------------------------


class TestGRPOStopToolConfig(TrlTestCase):
    """GRPOConfig.stop_tool_names field contract — no inference needed.

    Note: ``set_tmp_dir`` is a ``@pytest.fixture(autouse=True)`` on TrlTestCase;
    it runs automatically and sets ``self.tmp_dir`` before each test.
    """

    def test_stop_tool_names_default_none(self):
        assert _base_config(self.tmp_dir).stop_tool_names is None

    def test_stop_tool_names_single(self):
        cfg = _base_config(self.tmp_dir, stop_tool_names=["final_answer"])
        assert cfg.stop_tool_names == ["final_answer"]

    def test_stop_tool_names_multiple(self):
        cfg = _base_config(self.tmp_dir, stop_tool_names=["final_answer", "submit", "done"])
        assert cfg.stop_tool_names == ["final_answer", "submit", "done"]

    def test_stop_tool_names_empty_list(self):
        cfg = _base_config(self.tmp_dir, stop_tool_names=[])
        assert cfg.stop_tool_names == []

    def test_stop_tool_name_not_in_registered_tools_is_accepted(self):
        """Config accepts a stop tool name that has no matching registered tool."""
        cfg = _base_config(self.tmp_dir, stop_tool_names=["ghost_tool"])
        assert cfg.stop_tool_names == ["ghost_tool"]


# ---------------------------------------------------------------------------
# Training integration tests (require tool-call parsing support in transformers)
# ---------------------------------------------------------------------------


class TestGRPOStopToolTraining(TrlTestCase):
    """Integration tests verifying stop-tool loop termination during training.

    Each test mocks ``model.generate`` with a deterministic ``fake_generate``
    function so the assertions don't depend on what a tiny model happens to
    output — only on whether the training loop calls ``generate`` the right
    number of times and builds the right completion structure.
    """

    @pytest.mark.xfail(
        condition=Version(transformers.__version__) < Version("5.0.0"),
        reason="Tool parsing is not supported in transformers versions below 5.0.0",
        strict=True,
    )
    @require_jmespath
    def test_stop_tool_prevents_post_tool_generation(self):
        """When *all* samples call the stop tool, ``model.generate`` fires only once
        per training step (no post-tool generation turn).

        Without ``stop_tool_names``, TRL would enqueue a second generation turn
        for every sample that returned a tool result.  With the feature active,
        all samples are dropped from ``idxs_with_tool`` after the first iteration
        and the second ``generate`` call never happens.

        ``max_steps=1`` pins the run to a single training step so that the
        generate call count is deterministic regardless of dataset size.
        """
        tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
        final_answer_ids = _make_final_answer_ids(tokenizer)
        del tokenizer
        gc.collect()

        generate_call_count = [0]

        def fake_generate(input_ids, **kwargs):
            generate_call_count[0] += 1
            comp = torch.tensor(
                [final_answer_ids] * input_ids.shape[0],
                device=input_ids.device,
            )
            return torch.cat([input_ids, comp], dim=-1)

        dataset = load_dataset(_DATASET, "conversational_prompt_only", split="train")
        config = _base_config(self.tmp_dir, stop_tool_names=["final_answer"], max_steps=1)
        trainer = GRPOTrainer(
            model=_MODEL_ID,
            reward_funcs=_REWARD_MODEL_ID,
            args=config,
            train_dataset=dataset,
            tools=[multiply_tool, final_answer],
        )

        with patch.object(trainer.model, "generate", side_effect=fake_generate):
            trainer.train()

        assert generate_call_count[0] == 1, (
            f"model.generate was called {generate_call_count[0]} times; "
            "expected exactly 1 because all samples called the stop tool"
        )

    @pytest.mark.xfail(
        condition=Version(transformers.__version__) < Version("5.0.0"),
        reason="Tool parsing is not supported in transformers versions below 5.0.0",
        strict=True,
    )
    @require_jmespath
    def test_without_stop_tool_names_generation_continues(self):
        """Baseline: without ``stop_tool_names``, ``final_answer`` is treated as a
        normal tool and TRL re-enters the loop, calling ``model.generate`` more
        than once."""
        tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
        final_answer_ids = _make_final_answer_ids(tokenizer)
        done_ids = _make_done_ids(tokenizer)
        del tokenizer
        gc.collect()

        generate_call_count = [0]

        def fake_generate(input_ids, **kwargs):
            generate_call_count[0] += 1
            if generate_call_count[0] == 1:
                comp = torch.tensor(
                    [final_answer_ids] * input_ids.shape[0], device=input_ids.device
                )
            else:
                comp = torch.tensor(
                    [done_ids] * input_ids.shape[0], device=input_ids.device
                )
            return torch.cat([input_ids, comp], dim=-1)

        dataset = load_dataset(_DATASET, "conversational_prompt_only", split="train")
        config = _base_config(self.tmp_dir, stop_tool_names=None)
        trainer = GRPOTrainer(
            model=_MODEL_ID,
            reward_funcs=_REWARD_MODEL_ID,
            args=config,
            train_dataset=dataset,
            tools=[multiply_tool, final_answer],
        )

        with patch.object(trainer.model, "generate", side_effect=fake_generate):
            trainer.train()

        assert generate_call_count[0] > 1, (
            "Without stop_tool_names the loop must re-generate after the tool result"
        )

    @pytest.mark.xfail(
        condition=Version(transformers.__version__) < Version("5.0.0"),
        reason="Tool parsing is not supported in transformers versions below 5.0.0",
        strict=True,
    )
    @require_jmespath
    def test_mixed_stop_and_regular_tools(self):
        """When only *some* samples call the stop tool, generation continues
        only for the remaining samples.

        Setup (batch_size=3, max_steps=1 → exactly one training step):
          - samples 0 & 2: call ``final_answer`` (stop tool) → dropped after iter 1
          - sample 1: calls ``multiply_tool`` (regular) → gets a second generate turn

        Expected: ``model.generate`` is called exactly twice (initial + one
        post-tool turn for the non-stop sample).
        """
        tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
        final_answer_ids = _make_final_answer_ids(tokenizer)
        multiply_ids = _make_multiply_ids(tokenizer)
        done_ids = _make_done_ids(tokenizer)
        pad_id = tokenizer.pad_token_id or 0
        del tokenizer
        gc.collect()

        generate_call_count = [0]

        def fake_generate(input_ids, **kwargs):
            generate_call_count[0] += 1
            batch_size = input_ids.shape[0]
            if generate_call_count[0] == 1:
                # Interleave: sample 1 calls multiply_tool, others call final_answer.
                rows = [multiply_ids if i == 1 else final_answer_ids for i in range(batch_size)]
                max_len = max(len(r) for r in rows)
                padded = [r + [pad_id] * (max_len - len(r)) for r in rows]
                comp = torch.tensor(padded, device=input_ids.device)
            else:
                # Post-tool turn: only the multiply_tool sample remains.
                comp = torch.tensor([done_ids] * batch_size, device=input_ids.device)
            return torch.cat([input_ids, comp], dim=-1)

        dataset = load_dataset(_DATASET, "conversational_prompt_only", split="train")
        config = _base_config(self.tmp_dir, stop_tool_names=["final_answer"], max_steps=1)
        trainer = GRPOTrainer(
            model=_MODEL_ID,
            reward_funcs=_REWARD_MODEL_ID,
            args=config,
            train_dataset=dataset,
            tools=[multiply_tool, final_answer],
        )

        with patch.object(trainer.model, "generate", side_effect=fake_generate):
            trainer.train()

        assert generate_call_count[0] == 2, (
            f"model.generate was called {generate_call_count[0]} times; "
            "expected 2 (initial + one post-tool turn for the non-stop sample)"
        )

    @pytest.mark.xfail(
        condition=Version(transformers.__version__) < Version("5.0.0"),
        reason="Tool parsing is not supported in transformers versions below 5.0.0",
        strict=True,
    )
    @require_jmespath
    def test_stop_tool_completions_include_tool_result(self):
        """After the stop tool fires, the reward function must receive the
        ``role=tool`` message in the completion conversation.

        If ``completion_ids`` were not extended with the tool-result tokens,
        the decoded conversation would be missing that message entirely.
        """
        tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
        final_answer_ids = _make_final_answer_ids(tokenizer)
        del tokenizer
        gc.collect()

        captured_completions: list = []

        def capturing_reward(completions, **kwargs):
            captured_completions.extend(completions)
            return [1.0] * len(completions)

        def fake_generate(input_ids, **kwargs):
            comp = torch.tensor(
                [final_answer_ids] * input_ids.shape[0], device=input_ids.device
            )
            return torch.cat([input_ids, comp], dim=-1)

        dataset = load_dataset(_DATASET, "conversational_prompt_only", split="train")
        config = _base_config(self.tmp_dir, stop_tool_names=["final_answer"])
        trainer = GRPOTrainer(
            model=_MODEL_ID,
            reward_funcs=capturing_reward,
            args=config,
            train_dataset=dataset,
            tools=[multiply_tool, final_answer],
        )

        with patch.object(trainer.model, "generate", side_effect=fake_generate):
            trainer.train()

        assert captured_completions, "reward function was never called"

        for conv in captured_completions:
            tool_msgs = [m for m in conv if isinstance(m, dict) and m.get("role") == "tool"]
            assert tool_msgs, (
                "Expected at least one tool-result message in the completion; "
                "completion_ids may not have been extended with the tool-result tokens. "
                f"Conversation: {conv}"
            )
            for msg in tool_msgs:
                assert "name" in msg, f"tool message missing 'name' key: {msg}"
                assert "content" in msg, f"tool message missing 'content' key: {msg}"

    @pytest.mark.xfail(
        condition=Version(transformers.__version__) < Version("5.0.0"),
        reason="Tool parsing is not supported in transformers versions below 5.0.0",
        strict=True,
    )
    @require_jmespath
    def test_stop_tool_loss_is_finite(self):
        """Tool-result tokens must be masked out of the loss (``tool_mask=0``).

        If they were incorrectly included (``tool_mask=1``), the loss would be
        computed over non-model-generated tokens, which can produce NaN / Inf.
        We verify by checking that ``train_loss`` is a finite number.
        """
        tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
        final_answer_ids = _make_final_answer_ids(tokenizer)
        del tokenizer
        gc.collect()

        def fake_generate(input_ids, **kwargs):
            comp = torch.tensor(
                [final_answer_ids] * input_ids.shape[0], device=input_ids.device
            )
            return torch.cat([input_ids, comp], dim=-1)

        dataset = load_dataset(_DATASET, "conversational_prompt_only", split="train")
        config = _base_config(self.tmp_dir, stop_tool_names=["final_answer"], logging_steps=1)
        trainer = GRPOTrainer(
            model=_MODEL_ID,
            reward_funcs=_REWARD_MODEL_ID,
            args=config,
            train_dataset=dataset,
            tools=[multiply_tool, final_answer],
        )

        with patch.object(trainer.model, "generate", side_effect=fake_generate):
            trainer.train()

        train_loss = trainer.state.log_history[-1].get("train_loss")
        assert train_loss is not None
        assert train_loss == train_loss, "train_loss is NaN — tool_mask may be misconfigured"
        assert train_loss < float("inf"), "train_loss is Inf — tool_mask may be misconfigured"

    @pytest.mark.xfail(
        condition=Version(transformers.__version__) < Version("5.0.0"),
        reason="Tool parsing is not supported in transformers versions below 5.0.0",
        strict=True,
    )
    @require_jmespath
    def test_uncalled_stop_tool_is_noop(self):
        """A stop tool name that is never actually called has no effect on the loop.
        The loop continues normally via ``multiply_tool``.
        """
        tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
        multiply_ids = _make_multiply_ids(tokenizer)
        done_ids = _make_done_ids(tokenizer)
        del tokenizer
        gc.collect()

        generate_call_count = [0]

        def fake_generate(input_ids, **kwargs):
            generate_call_count[0] += 1
            if generate_call_count[0] == 1:
                comp = torch.tensor(
                    [multiply_ids] * input_ids.shape[0], device=input_ids.device
                )
            else:
                comp = torch.tensor(
                    [done_ids] * input_ids.shape[0], device=input_ids.device
                )
            return torch.cat([input_ids, comp], dim=-1)

        dataset = load_dataset(_DATASET, "conversational_prompt_only", split="train")
        # "final_answer" is listed but the model only calls "multiply_tool".
        config = _base_config(self.tmp_dir, stop_tool_names=["final_answer"])
        trainer = GRPOTrainer(
            model=_MODEL_ID,
            reward_funcs=_REWARD_MODEL_ID,
            args=config,
            train_dataset=dataset,
            tools=[multiply_tool, final_answer],
        )

        with patch.object(trainer.model, "generate", side_effect=fake_generate):
            trainer.train()

        assert generate_call_count[0] > 1, (
            "Loop should continue because the stop tool was never called"
        )

    @pytest.mark.xfail(
        condition=Version(transformers.__version__) < Version("5.0.0"),
        reason="Tool parsing is not supported in transformers versions below 5.0.0",
        strict=True,
    )
    @require_jmespath
    def test_stop_tool_compatible_with_max_tool_calling_iterations_one(self):
        """``max_tool_calling_iterations=1`` and ``stop_tool_names`` are compatible;
        training completes without errors."""
        tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
        final_answer_ids = _make_final_answer_ids(tokenizer)
        del tokenizer
        gc.collect()

        def fake_generate(input_ids, **kwargs):
            comp = torch.tensor(
                [final_answer_ids] * input_ids.shape[0], device=input_ids.device
            )
            return torch.cat([input_ids, comp], dim=-1)

        dataset = load_dataset(_DATASET, "conversational_prompt_only", split="train")
        config = _base_config(
            self.tmp_dir,
            stop_tool_names=["final_answer"],
            max_tool_calling_iterations=1,
        )
        trainer = GRPOTrainer(
            model=_MODEL_ID,
            reward_funcs=_REWARD_MODEL_ID,
            args=config,
            train_dataset=dataset,
            tools=[multiply_tool, final_answer],
        )

        with patch.object(trainer.model, "generate", side_effect=fake_generate):
            trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
