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

import asyncio
import itertools
import math
import queue
import types

import pytest
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.testing_utils import torch_device

from trl.experimental.async_distillation import AsyncDistillationConfig, AsyncDistillationTrainer
from trl.experimental.async_distillation.async_distillation_trainer import (
    DataCollatorForRollout,
    FixedCountBatcher,
    TokenBudgetBatcher,
    _add_tail_bucket,
    _balance_by_squared_length,
    _jsd_divergence,
    _narrow_top1_actual_support,
)
from trl.experimental.async_distillation.async_rollout_worker import (
    RolloutSample,
    _AsyncRolloutLoop,
    _parse_teacher_logprobs_at_position,
)
from trl.experimental.async_distillation.vllm_client import VLLMClient
from trl.experimental.server_distillation.server_distillation_trainer import (
    _jsd_divergence as _reference_jsd_divergence,
)

from ..testing_utils import TrlTestCase, is_ampere_or_newer


TEACHER_TOP_K = 4


def _make_teacher_topk(
    vocab_size: int, actual_token_ids: list[int], top_k: int, seed: int
) -> tuple[list[list[int]], list[list[float]]]:
    """Build synthetic per-position teacher candidates, shaped like what `_score_with_teacher` returns.

    Always includes the actual/realized token, matching the real server's guarantee that it's reported regardless of
    rank; otherwise `_narrow_top1_actual_support`'s `beta != 0.0` path would never find it among random candidates and
    silently mask every position out of the loss.
    """
    generator = torch.Generator().manual_seed(seed)
    ids, logprobs = [], []
    for actual_id in actual_token_ids:
        logits = torch.randn(vocab_size, generator=generator)
        log_probs = torch.log_softmax(logits, dim=-1)
        top_logprobs, top_ids = log_probs.topk(top_k)
        top_ids_list, top_logprobs_list = top_ids.tolist(), top_logprobs.tolist()
        if actual_id not in top_ids_list:
            top_ids_list[-1] = actual_id
            top_logprobs_list[-1] = log_probs[actual_id].item()
        ids.append(top_ids_list)
        logprobs.append(top_logprobs_list)
    return ids, logprobs


class _StubRolloutWorker:
    """Minimal rollout worker stub for testing the trainer in isolation, bypassing real vLLM/network."""

    def __init__(self, tokenizer, dataset, vocab_size: int, samples_per_weight_sync: int = 10):
        self.rollout_buffer = queue.Queue()
        self._samples_per_weight_sync = samples_per_weight_sync
        self._model_version = 0
        self._vocab_size = vocab_size
        self._sample_iter = self._make_sample_iter(tokenizer, dataset)

    def _make_sample_iter(self, tokenizer, dataset):
        for idx, row in enumerate(itertools.cycle(dataset)):
            completion = row["completion"]
            prompt_ids = tokenizer.apply_chat_template(
                row["prompt"], tokenize=True, add_generation_prompt=True, return_dict=False
            )
            prompt_completion_ids = tokenizer.apply_chat_template(
                row["prompt"] + completion, tokenize=True, add_generation_prompt=False, return_dict=False
            )
            completion_ids = prompt_completion_ids[len(prompt_ids) :]
            teacher_topk_ids, teacher_topk_logprobs = _make_teacher_topk(
                self._vocab_size, completion_ids, TEACHER_TOP_K, seed=idx
            )
            yield RolloutSample(
                prompt=row["prompt"],
                completion=completion,
                input_ids=prompt_ids + completion_ids,
                completion_mask=[0] * len(prompt_ids) + [1] * len(completion_ids),
                teacher_topk_ids=[[] for _ in prompt_ids] + teacher_topk_ids,
                teacher_topk_logprobs=[[] for _ in prompt_ids] + teacher_topk_logprobs,
                model_version=self._model_version,
                teacher_id="default",
                metrics={"rollout_time_ms": 1.0},
            )

    def _fill_queue(self):
        for _ in range(self._samples_per_weight_sync):
            self.rollout_buffer.put(next(self._sample_iter))

    def start(self):
        self._fill_queue()

    def update_model_version(self, version):
        self._model_version = version
        self._fill_queue()

    def stop(self):
        pass

    def check_health(self, stale_after_s):
        pass


class _StubWeightTransfer:
    """No-op weight transfer for testing the trainer without a real vLLM server."""

    def init_weight_transfer(self):
        pass

    def pause(self):
        pass

    def send_weights(self, iterator):
        for _ in iterator:  # drain the param stream like the real client does
            pass

    def resume(self):
        pass

    def destroy(self):
        pass


def _rollout_sample(length: int, top_k: int = TEACHER_TOP_K) -> dict:
    # First token is a prompt token (completion_mask 0, empty teacher candidates); the rest are completion tokens.
    teacher_topk_ids = [[]] + [list(range(top_k))] * (length - 1)
    teacher_topk_logprobs = [[]] + [[-float(i) - 0.1 for i in range(top_k)]] * (length - 1)
    return {
        "input_ids": list(range(length)),
        "completion_mask": [0] + [1] * (length - 1),
        "teacher_topk_ids": teacher_topk_ids,
        "teacher_topk_logprobs": teacher_topk_logprobs,
        "metrics": {"rollout_time_ms": 1.0},
    }


class TestPackingAwareBatching:
    """Packing/collation is pure scheduling/tensorization (ported from async_grpo's batchers, unchanged), so these
    run without a GPU. The teacher-candidate padding in the collator is new and is the focus here.
    """

    def test_fixed_count_batcher_yields_balanced_fixed_count_micro_batches(self):
        source = (_rollout_sample(length) for length in itertools.cycle((4, 3, 2, 1)))
        batcher = FixedCountBatcher(source, num_processes=2, microbatch_size=4)

        micro_batches = list(itertools.islice(iter(batcher), 3))
        assert len(micro_batches) == 3
        for groups in micro_batches:
            assert len(groups) == 2
            assert all(len(group) > 0 for group in groups)
            assert sum(len(group) for group in groups) == 4

    def test_token_budget_batcher_respects_budget_and_fills_every_row(self):
        source = (_rollout_sample(3) for _ in range(100))
        batcher = TokenBudgetBatcher(source, num_processes=2, token_budget=8)

        micro_batches = list(itertools.islice(iter(batcher), 5))
        assert len(micro_batches) == 5
        for groups in micro_batches:
            assert len(groups) == 2
            assert all(len(group) > 0 for group in groups)
            assert all(sum(len(s["input_ids"]) for s in group) <= 8 for group in groups)

    def test_balance_equalizes_squared_length(self):
        samples = [_rollout_sample(length) for length in (3, 3, 2, 2)]
        groups = _balance_by_squared_length(samples, num_groups=2)
        loads = [sum(len(s["input_ids"]) ** 2 for s in group) for group in groups]
        assert loads[0] == loads[1]

    def test_collator_pads_teacher_candidates_and_masks_prompt_positions(self):
        collator = DataCollatorForRollout(pad_token_id=0, teacher_top_k=TEACHER_TOP_K, num_processes=1)
        a = _rollout_sample(3)  # input_ids [0, 1, 2]; position 0 is prompt (no candidates)
        batch = collator([[[a]]])

        width = (
            TEACHER_TOP_K + 1
        )  # +1 for the sentinel-padded slot (no tail bucket added here; that's compute_loss's job)
        assert batch["teacher_topk_ids"].shape == (1, 3, width)
        assert batch["teacher_topk_logprobs"].shape == (1, 3, width)
        # Prompt position (index 0): every candidate slot is the -1/-inf sentinel.
        assert batch["teacher_topk_ids"][0, 0].tolist() == [-1] * width
        assert torch.isinf(batch["teacher_topk_logprobs"][0, 0]).all()
        # Completion positions (index 1, 2): the real top_k candidates survive, the extra slot is padding.
        assert batch["teacher_topk_ids"][0, 1, :TEACHER_TOP_K].tolist() == list(range(TEACHER_TOP_K))
        assert batch["teacher_topk_ids"][0, 1, TEACHER_TOP_K] == -1

    def test_collator_pads_unequal_rows(self):
        collator = DataCollatorForRollout(pad_token_id=0, teacher_top_k=TEACHER_TOP_K, num_processes=2)
        a = _rollout_sample(3)  # input_ids [0, 1, 2]
        b = _rollout_sample(2)  # input_ids [0, 1]

        batch = collator([[[a], [b]]])

        assert batch["input_ids"].tolist() == [[0, 1, 2], [0, 1, 0]]  # row b right-padded with pad_token_id
        assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
        assert batch["completion_mask"].tolist() == [[0, 1, 1], [0, 1, 0]]
        assert batch["global_n_tokens"].tolist() == [3.0, 3.0]  # a: 2 + b: 1 completion tokens


class TestParseTeacherLogprobs:
    """`/get_sequence_logprobs/` returns already rank-sorted parallel lists; parsing here is just dropping NaNs."""

    def test_passes_through_already_sorted_candidates(self):
        ids, logprobs = _parse_teacher_logprobs_at_position([-0.1, -1.0, -2.0], [5, 1, 9])
        assert ids == [5, 1, 9]
        assert logprobs == pytest.approx([-0.1, -1.0, -2.0])

    def test_drops_nan_entries(self):
        # The server emits `None` for a NaN logprob (see trl/scripts/vllm_serve.py); such entries carry no signal.
        ids, logprobs = _parse_teacher_logprobs_at_position([-0.1, None, -2.0], [5, 1, 9])
        assert ids == [5, 9]
        assert logprobs == pytest.approx([-0.1, -2.0])

    def test_empty_position_returns_empty(self):
        # A position with no teacher logprobs at all (e.g. `prompt_logprobs[pos] is None` server-side) comes back
        # as an empty pair of lists, same convention as a masked-out prompt position.
        ids, logprobs = _parse_teacher_logprobs_at_position([], [])
        assert ids == [] and logprobs == []


def _bare_loop(tokenizer, teacher_server_urls):
    # _generate_and_score_one/_score_with_teacher only read the attributes set below off self, so we skip the heavy
    # __init__ (mp.Queue/mp.Value/mp.Event, child-process bookkeeping) and set just those, mirroring
    # AsyncGRPOTrainer's own `_bare_loop` test helper.
    loop = object.__new__(_AsyncRolloutLoop)
    loop.tokenizer = tokenizer
    loop.chat_template_kwargs = {}
    loop.teacher_server_urls = teacher_server_urls
    loop.teacher_top_k = 4
    loop.teacher_temperature = 1.0
    loop.request_timeout = 30
    loop._model_version_value = types.SimpleNamespace(value=0)
    return loop


class TestMultiTeacherRouting:
    """A single configured teacher scores every row regardless of `teacher_id` (plain on-policy distillation).

    Multiple teachers enable MOPD: each row's `teacher_id` column selects among them; a missing or unmapped
    `teacher_id` is a configuration error, not a silent fallback.
    """

    def _run(self, row, teacher_server_urls, posted_urls):
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        loop = _bare_loop(tokenizer, teacher_server_urls)

        async def fake_generate_one_turn(prompt_ids):
            return [7]

        loop._generate_one_turn = fake_generate_one_turn

        async def fake_post(base_url, path, payload, timeout):
            posted_urls.append(base_url)
            return {"logprobs": [[[0.0]]], "logprob_token_ids": [[[7]]]}

        loop._post = fake_post
        return asyncio.run(loop._generate_and_score_one(row))

    def test_single_teacher_ignores_teacher_id(self):
        posted_urls = []
        self._run(
            {"prompt": [{"role": "user", "content": "hi"}], "teacher_id": "whatever"},
            {"default": "http://default:8001"},
            posted_urls,
        )
        assert posted_urls == ["http://default:8001"]

    def test_single_teacher_works_with_no_teacher_id_column(self):
        posted_urls = []
        self._run(
            {"prompt": [{"role": "user", "content": "hi"}]},
            {"default": "http://default:8001"},
            posted_urls,
        )
        assert posted_urls == ["http://default:8001"]

    def test_routes_to_matching_teacher_id(self):
        posted_urls = []
        self._run(
            {"prompt": [{"role": "user", "content": "hi"}], "teacher_id": "math"},
            {"math": "http://math:8002", "code": "http://code:8003"},
            posted_urls,
        )
        assert posted_urls == ["http://math:8002"]

    def test_multi_teacher_raises_on_missing_teacher_id(self):
        with pytest.raises(KeyError, match="teacher_id=None"):
            self._run(
                {"prompt": [{"role": "user", "content": "hi"}]},
                {"math": "http://math:8002", "code": "http://code:8003"},
                [],
            )

    def test_multi_teacher_raises_on_unmapped_teacher_id(self):
        with pytest.raises(KeyError, match="teacher_id='unknown'"):
            self._run(
                {"prompt": [{"role": "user", "content": "hi"}], "teacher_id": "unknown"},
                {"math": "http://math:8002", "code": "http://code:8003"},
                [],
            )


class TestBetaRange:
    """beta must be a valid generalized-JSD interpolation coefficient in [0.0, 1.0], matching
    `DistillationTrainer`/`GKDConfig`'s own bounds; values outside that range are never meaningful.
    """

    @pytest.mark.parametrize("beta", [-0.1, 1.1, 2.0])
    def test_out_of_range_beta_raises(self, beta):
        # use_cpu/bf16 avoid an unrelated `TrainingArguments` validation error on machines without a bf16-capable
        # GPU; irrelevant to what this test checks.
        with pytest.raises(ValueError, match=r"beta must be in \[0.0, 1.0\]"):
            AsyncDistillationConfig(output_dir="/tmp/unused", report_to="none", use_cpu=True, bf16=False, beta=beta)


class TestNarrowTop1ActualSupport:
    """`beta != 0.0` narrows the teacher-reported support to just the candidates the divergence needs, mirroring
    `ServerDistillationTrainer`'s `beta > 0` path exactly, including its `beta == 1.0` special case: for `0 < beta <
    1`, the support is the teacher's own top-1 token plus the completion's actual/realized token (the only two
    identities the wire protocol guarantees a teacher logprob for); at `beta == 1.0` (pure reverse KL, purely
    student-weighted), the teacher's top-1 token is dropped entirely and only the actual token remains.
    """

    def test_top1_and_actual_both_valid_and_distinct(self):
        # position 0: top-1 is token 5, actual (realized) token is 9, found at index 2 of the reported candidates.
        teacher_topk_ids = torch.tensor([[[5, 3, 9, -1]]])
        teacher_topk_logprobs = torch.tensor([[[-0.1, -1.0, -2.0, float("-inf")]]])
        actual_token_ids = torch.tensor([[9]])
        valid_mask_wide = teacher_topk_ids != -1

        narrow_ids, narrow_logps, mask = _narrow_top1_actual_support(
            teacher_topk_ids, teacher_topk_logprobs, actual_token_ids, valid_mask_wide, beta=0.5
        )
        assert narrow_ids.tolist() == [[[5, 9]]]
        assert narrow_logps.squeeze().tolist() == pytest.approx([-0.1, -2.0])
        assert mask.tolist() == [[[True, True]]]

    def test_actual_token_same_as_top1_is_deduplicated(self):
        # The realized token IS the teacher's top-1 choice: the second (actual) slot must be masked out, not
        # double-counted, since it's the same token as the first (top-1) slot.
        teacher_topk_ids = torch.tensor([[[5, 3, 9, -1]]])
        teacher_topk_logprobs = torch.tensor([[[-0.1, -1.0, -2.0, float("-inf")]]])
        actual_token_ids = torch.tensor([[5]])
        valid_mask_wide = teacher_topk_ids != -1

        narrow_ids, _, mask = _narrow_top1_actual_support(
            teacher_topk_ids, teacher_topk_logprobs, actual_token_ids, valid_mask_wide, beta=0.5
        )
        assert narrow_ids.tolist() == [[[5, 5]]]
        assert mask.tolist() == [[[True, False]]]

    def test_actual_token_missing_from_reported_candidates(self):
        # The realized token never appears among the teacher's reported candidates at all (e.g. a low-probability
        # token the server didn't report); the actual slot must be masked out, not silently given a wrong logprob.
        teacher_topk_ids = torch.tensor([[[5, 3, -1, -1]]])
        teacher_topk_logprobs = torch.tensor([[[-0.1, -1.0, float("-inf"), float("-inf")]]])
        actual_token_ids = torch.tensor([[42]])
        valid_mask_wide = teacher_topk_ids != -1

        narrow_ids, _, mask = _narrow_top1_actual_support(
            teacher_topk_ids, teacher_topk_logprobs, actual_token_ids, valid_mask_wide, beta=0.5
        )
        assert narrow_ids.tolist() == [[[5, 42]]]
        assert mask.tolist() == [[[True, False]]]

    def test_top1_missing_but_actual_present(self):
        # No usable teacher signal at all for the top-1 slot (e.g. every candidate was NaN server-side), but the
        # actual token still happens to be present among the (otherwise-invalid) reported candidates.
        teacher_topk_ids = torch.tensor([[[-1, 9, -1, -1]]])
        teacher_topk_logprobs = torch.tensor([[[float("-inf"), -2.0, float("-inf"), float("-inf")]]])
        actual_token_ids = torch.tensor([[9]])
        valid_mask_wide = teacher_topk_ids != -1

        _, _, mask = _narrow_top1_actual_support(
            teacher_topk_ids, teacher_topk_logprobs, actual_token_ids, valid_mask_wide, beta=0.5
        )
        assert mask.tolist() == [[[False, True]]]

    def test_beta_one_drops_teacher_top1_entirely(self):
        # Pure reverse KL is purely student-weighted: the teacher's top-1 token (5) must not appear in the support
        # at all, even though it's valid and distinct from the actual token (9) - only width-1 support remains.
        teacher_topk_ids = torch.tensor([[[5, 3, 9, -1]]])
        teacher_topk_logprobs = torch.tensor([[[-0.1, -1.0, -2.0, float("-inf")]]])
        actual_token_ids = torch.tensor([[9]])
        valid_mask_wide = teacher_topk_ids != -1

        narrow_ids, narrow_logps, mask = _narrow_top1_actual_support(
            teacher_topk_ids, teacher_topk_logprobs, actual_token_ids, valid_mask_wide, beta=1.0
        )
        assert narrow_ids.shape[-1] == 1
        assert narrow_ids.tolist() == [[[9]]]
        assert narrow_logps.squeeze().item() == pytest.approx(-2.0)
        assert mask.tolist() == [[[True]]]

    def test_beta_one_actual_token_missing(self):
        # Pure reverse KL, but the actual token isn't among the teacher's reported candidates: the lone support
        # slot must be masked out (no usable teacher signal at all), not silently wrong.
        teacher_topk_ids = torch.tensor([[[5, 3, -1, -1]]])
        teacher_topk_logprobs = torch.tensor([[[-0.1, -1.0, float("-inf"), float("-inf")]]])
        actual_token_ids = torch.tensor([[42]])
        valid_mask_wide = teacher_topk_ids != -1

        narrow_ids, _, mask = _narrow_top1_actual_support(
            teacher_topk_ids, teacher_topk_logprobs, actual_token_ids, valid_mask_wide, beta=1.0
        )
        assert narrow_ids.tolist() == [[[42]]]
        assert mask.tolist() == [[[False]]]

    def test_narrow_support_feeds_add_tail_bucket_and_jsd_divergence_without_error(self):
        # End-to-end sanity: the narrow support's shape/dtype/mask conventions must be exactly what
        # `_add_tail_bucket`/`_jsd_divergence` (shared with the beta=0.0 path) expect.
        teacher_topk_ids = torch.tensor([[[5, 3, 9, -1], [7, -1, -1, -1]]])
        teacher_topk_logprobs = torch.tensor(
            [[[-0.1, -1.0, -2.0, float("-inf")], [-0.05, float("-inf"), float("-inf"), float("-inf")]]]
        )
        actual_token_ids = torch.tensor([[9, 7]])
        valid_mask_wide = teacher_topk_ids != -1

        narrow_ids, teacher_logps, mask = _narrow_top1_actual_support(
            teacher_topk_ids, teacher_topk_logprobs, actual_token_ids, valid_mask_wide, beta=0.5
        )
        student_logps_full = torch.log_softmax(torch.randn(1, 2, 32), dim=-1)
        student_logps = student_logps_full.gather(-1, narrow_ids.clamp_min(0))
        neg_inf = torch.full((), float("-inf"))
        student_logps = torch.where(mask, student_logps, neg_inf)
        teacher_logps = torch.where(mask, teacher_logps, neg_inf)

        student_full, support_mask = _add_tail_bucket(student_logps, mask)
        teacher_full, _ = _add_tail_bucket(teacher_logps, mask)
        jsd = _jsd_divergence(student_full, teacher_full, beta=0.5, support_mask=support_mask)
        assert torch.isfinite(jsd).all()
        # Individual per-slot terms can be negative (e.g. student_prob(x) * (log student(x) - log teacher(x)) when
        # student(x) < teacher(x)); only the per-token sum across the support dimension (the actual KL/JSD) is
        # guaranteed non-negative (Gibbs' inequality).
        assert (jsd.sum(dim=-1) >= -1e-5).all()

    def test_narrow_support_beta_one_feeds_add_tail_bucket_and_jsd_divergence_without_error(self):
        # Same end-to-end sanity check, but for the width-1 beta==1.0 support.
        teacher_topk_ids = torch.tensor([[[5, 3, 9, -1], [7, -1, -1, -1]]])
        teacher_topk_logprobs = torch.tensor(
            [[[-0.1, -1.0, -2.0, float("-inf")], [-0.05, float("-inf"), float("-inf"), float("-inf")]]]
        )
        actual_token_ids = torch.tensor([[9, 7]])
        valid_mask_wide = teacher_topk_ids != -1

        narrow_ids, teacher_logps, mask = _narrow_top1_actual_support(
            teacher_topk_ids, teacher_topk_logprobs, actual_token_ids, valid_mask_wide, beta=1.0
        )
        student_logps_full = torch.log_softmax(torch.randn(1, 2, 32), dim=-1)
        student_logps = student_logps_full.gather(-1, narrow_ids.clamp_min(0))
        neg_inf = torch.full((), float("-inf"))
        student_logps = torch.where(mask, student_logps, neg_inf)
        teacher_logps = torch.where(mask, teacher_logps, neg_inf)

        student_full, support_mask = _add_tail_bucket(student_logps, mask)
        teacher_full, _ = _add_tail_bucket(teacher_logps, mask)
        jsd = _jsd_divergence(student_full, teacher_full, beta=1.0, support_mask=support_mask)
        assert torch.isfinite(jsd).all()
        # Individual per-slot terms can be negative (e.g. student_prob(x) * (log student(x) - log teacher(x)) when
        # student(x) < teacher(x)); only the per-token sum across the support dimension (the actual KL/JSD) is
        # guaranteed non-negative (Gibbs' inequality).
        assert (jsd.sum(dim=-1) >= -1e-5).all()

    def test_padded_positions_keep_forward_and_backward_finite(self):
        # Adapted from ServerDistillationTrainer's own `TestServerReverseKLPaddingMask` test: a ragged batch where
        # one sample's completion is shorter than another's, so some positions are all -inf sentinels (masked out
        # via has_teacher_signal in compute_loss, but this checks the loss math itself, given already-neutralized
        # padding, never leaks -inf/NaN into the backward pass).
        teacher_topk_ids = torch.tensor([[[5, 3, 9, -1], [-1, -1, -1, -1]], [[5, 3, 9, -1], [7, 2, 4, -1]]])
        teacher_topk_logprobs = torch.tensor(
            [
                [[-0.1, -1.0, -2.0, float("-inf")], [float("-inf")] * 4],
                [[-0.1, -1.0, -2.0, float("-inf")], [-0.05, -1.2, -2.5, float("-inf")]],
            ]
        )
        actual_token_ids = torch.tensor([[9, 0], [9, 7]])
        valid_mask_wide = teacher_topk_ids != -1

        for beta in (0.5, 1.0):
            narrow_ids, teacher_logps, mask = _narrow_top1_actual_support(
                teacher_topk_ids, teacher_topk_logprobs, actual_token_ids, valid_mask_wide, beta=beta
            )
            raw_student = torch.randn(2, 2, 32, requires_grad=True)
            student_log_probs_full = torch.log_softmax(raw_student, dim=-1)
            student_logps = student_log_probs_full.gather(-1, narrow_ids.clamp_min(0))
            neg_inf = torch.full((), float("-inf"))
            student_logps = torch.where(mask, student_logps, neg_inf)
            teacher_logps = torch.where(mask, teacher_logps, neg_inf)

            student_full, support_mask = _add_tail_bucket(student_logps, mask)
            teacher_full, _ = _add_tail_bucket(teacher_logps, mask)
            jsd = _jsd_divergence(student_full, teacher_full, beta=beta, support_mask=support_mask)
            assert torch.isfinite(jsd).all(), f"beta={beta}"

            # The fully-masked position (sample 0, position 1) must not contribute -inf/NaN gradient once summed
            # and backpropagated, mirroring compute_loss's has_teacher_signal exclusion of such positions.
            has_teacher_signal = mask.any(dim=-1)
            loss = jsd.sum(dim=-1)[has_teacher_signal].sum()
            loss.backward()
            assert torch.isfinite(raw_student.grad).all(), f"beta={beta}"


class TestSparseGeneralizedJSDMatchesFullVocabReference:
    """Compares our sparse top-k pipeline (`_add_tail_bucket` + `_jsd_divergence`) against a plain dense-vocabulary
    generalized-JSD computation — `ServerDistillationTrainer`'s own `_jsd_divergence`, called with `support_mask=None`
    so it takes its dense `F.kl_div` branch instead of the masked/sparse one. Feeding our pipeline the *full*
    vocabulary as its "top-k" support (`top_k = vocab_size`) leaves ~0 residual mass for the tail bucket, so the two
    must match up to that negligible epsilon, not just approximately.
    """

    def test_forward_kl_matches_full_vocab_generalized_jsd_loss(self):
        torch.manual_seed(0)
        batch, seq_len, vocab_size = 2, 5, 32
        student_logits = torch.randn(batch, seq_len, vocab_size)
        teacher_logits = torch.randn(batch, seq_len, vocab_size)

        student_log_probs_full = torch.log_softmax(student_logits, dim=-1)
        teacher_log_probs_full = torch.log_softmax(teacher_logits, dim=-1)

        reference_jsd = _reference_jsd_divergence(student_log_probs_full, teacher_log_probs_full, beta=0.0)
        reference = reference_jsd.sum() / (batch * seq_len)  # batchmean, matching this test's own reduction below

        # Simulate what the async worker/collator produce, but with the full vocabulary as the "top-k" support so
        # nothing is truncated (see class docstring for why this is a valid dense reference).
        teacher_topk_logprobs, teacher_topk_ids = teacher_log_probs_full.topk(vocab_size, dim=-1)

        student_support_logps = student_log_probs_full.gather(-1, teacher_topk_ids)
        valid_mask = torch.ones_like(teacher_topk_ids, dtype=torch.bool)

        student_log_probs, support_mask = _add_tail_bucket(student_support_logps, valid_mask)
        teacher_log_probs, _ = _add_tail_bucket(teacher_topk_logprobs, valid_mask)

        jsd = _jsd_divergence(student_log_probs, teacher_log_probs, beta=0.0, support_mask=support_mask)
        ours = jsd.sum() / max(jsd.shape[0] * jsd.shape[1], 1)  # batchmean, matching the reference's reduction

        torch.testing.assert_close(ours, reference)


@pytest.mark.skipif(
    not is_ampere_or_newer() and torch_device != "xpu",
    reason="Flash Attention 2 requires Ampere or newer GPU, or XPU",
)
class TestAsyncDistillationTrainer(TrlTestCase):
    def test_init_minimal(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        AsyncDistillationTrainer(
            model=model_id,
            train_dataset=dataset,
            rollout_worker=_StubRolloutWorker(tokenizer, dataset, vocab_size=tokenizer.vocab_size),
            weight_transfer=_StubWeightTransfer(),
        )

    def test_train(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        training_args = AsyncDistillationConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,
            teacher_top_k=TEACHER_TOP_K,
            max_completion_length=8,
            vllm_server_timeout=5.0,  # short timeout so test fails fast if queue runs dry
            token_budget=-1,  # disable token budgeting (no real vLLM server to query max_model_len from)
            report_to="none",
        )
        trainer = AsyncDistillationTrainer(
            model=model_id,
            args=training_args,
            train_dataset=dataset,
            rollout_worker=_StubRolloutWorker(tokenizer, dataset, vocab_size=tokenizer.vocab_size),
            weight_transfer=_StubWeightTransfer(),
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_teacher_vocab_size_mismatch_raises(self, monkeypatch):
        # compute_loss gathers the teacher's reported token ids directly against the student's logits, so a
        # vocab_size mismatch would silently score garbage rather than error — mirrors DistillationTrainer's
        # local-teacher `test_teacher_vocab_size_mismatch_raises`, adapted for a remote, HTTP-served teacher: no
        # rollout_worker is injected here, so the real (non-stub) path that queries the teacher's `/v1/models` and
        # checks its vocab_size actually runs.
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")
        monkeypatch.setattr(VLLMClient, "get_model_id", lambda self: "trl-internal-testing/tiny-LlamaForCausalLM-3.2")

        training_args = AsyncDistillationConfig(
            output_dir=self.tmp_dir,
            teacher_server_urls={"default": "http://localhost:8001"},
            report_to="none",
        )
        with pytest.raises(ValueError, match="vocab_size"):
            AsyncDistillationTrainer(model=model_id, args=training_args, train_dataset=dataset)

    def test_teacher_vocab_size_mismatch_identifies_teacher(self, monkeypatch):
        # MOPD (multi-teacher on-policy distillation): the mismatch check must run per teacher and name the one
        # that actually mismatches, not just whichever it happens to check first.
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")

        def fake_get_model_id(self):
            return "trl-internal-testing/tiny-LlamaForCausalLM-3.2" if "8002" in self.server_url else model_id

        monkeypatch.setattr(VLLMClient, "get_model_id", fake_get_model_id)

        training_args = AsyncDistillationConfig(
            output_dir=self.tmp_dir,
            teacher_server_urls={"math": "http://localhost:8001", "code": "http://localhost:8002"},
            report_to="none",
        )
        with pytest.raises(ValueError, match="'code'"):
            AsyncDistillationTrainer(model=model_id, args=training_args, train_dataset=dataset)

    @pytest.mark.parametrize("beta", [0.25, 0.5, 0.75, 1.0])
    def test_train_with_nonzero_beta(self, beta):
        # Adapted from ServerDistillationTrainer's own `test_reverse_kl_finite_grad_with_ragged_batch`: a real
        # training run through the narrow top-1 + actual-token support path (including the beta==1.0 special case),
        # checking grad_norm/loss stay finite throughout, not just that training completes.
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        training_args = AsyncDistillationConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            teacher_top_k=TEACHER_TOP_K,
            max_completion_length=8,
            vllm_server_timeout=5.0,
            token_budget=-1,
            beta=beta,
            report_to="none",
        )
        trainer = AsyncDistillationTrainer(
            model=model_id,
            args=training_args,
            train_dataset=dataset,
            rollout_worker=_StubRolloutWorker(tokenizer, dataset, vocab_size=tokenizer.vocab_size),
            weight_transfer=_StubWeightTransfer(),
        )
        trainer.train()

        log_history = [rec for rec in trainer.state.log_history if "grad_norm" in rec]
        assert log_history, "Expected at least one grad_norm log entry during training"
        for record in log_history:
            assert math.isfinite(record["grad_norm"]), f"grad_norm={record['grad_norm']} leaked -inf into backward"
            assert math.isfinite(record["loss"])
