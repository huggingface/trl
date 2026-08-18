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
import multiprocessing as mp
import queue
import types
from collections import defaultdict

import pytest
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.testing_utils import torch_device

from trl.experimental.async_distillation import AsyncDistillationConfig, AsyncDistillationTrainer
from trl.experimental.async_distillation.async_distillation_trainer import (
    DataCollatorForRollout,
    FixedCountBatcher,
    RolloutWorkerProtocol,
    TokenBudgetBatcher,
    _add_tail_bucket,
    _balance_by_squared_length,
    _jsd_divergence,
    _narrow_top1_actual_support,
    _reduce_metric,
)
from trl.experimental.async_distillation.async_rollout_worker import (
    RolloutSample,
    _AsyncRolloutLoop,
    _parse_teacher_logprobs_at_position,
)
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
        self.metrics_queue = queue.Queue()  # drained by the trainer in `log()`; this stub measures nothing
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
                metrics={},
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


def _rollout_sample(length: int, top_k: int = TEACHER_TOP_K, teacher_id: str = "default") -> dict:
    # First token is a prompt token (completion_mask 0, empty teacher candidates); the rest are completion tokens.
    teacher_topk_ids = [[]] + [list(range(top_k))] * (length - 1)
    teacher_topk_logprobs = [[]] + [[-float(i) - 0.1 for i in range(top_k)]] * (length - 1)
    return {
        "input_ids": list(range(length)),
        "completion_mask": [0] + [1] * (length - 1),
        "teacher_topk_ids": teacher_topk_ids,
        "teacher_topk_logprobs": teacher_topk_logprobs,
        "teacher_id": teacher_id,
        "metrics": {},
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
        batcher = TokenBudgetBatcher(source, num_processes=2, token_budget=8, metrics=defaultdict(list))

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
        assert batch["global_n_forward_tokens"].tolist() == [5.0, 5.0]  # every token forwarded, prompts included
        assert batch["mean_seq_len"].tolist() == [2.5, 2.5]

        # Sample and packing metrics are aggregated here on rank 0, not broadcast with the batch and reduced back.
        assert collator.metrics["sample/forwarded_tokens_mean"] == [2.5]
        assert collator.metrics["sample/trained_tokens_mean"] == [1.5]
        assert collator.metrics["batch/masked_token_frac"] == [(2, 5)]  # the two prompt tokens earn no gradient
        assert collator.metrics["batch/samples_per_row"] == [1.0]
        assert collator.metrics["batch/row_tokens_max"] == [3.0]
        assert collator.metrics["batch/row_imbalance"] == [9 / 6.5]  # Σ Lᵢ² of 9 and 4, against their mean
        assert collator.metrics["batch/pad_frac"] == [(1, 6)]  # the one slot row b was padded with
        assert "batch/row_fill_frac" not in collator.metrics  # no budget passed, so nothing to fill

    def test_collator_packs_teacher_id_per_token_for_mopd(self):
        # Packing concatenates samples from different teachers into one row, so compute_loss can only attribute a
        # token's jsd/teacher_entropy to a teacher if the id is broadcast across that sample's own tokens here.
        collator = DataCollatorForRollout(
            pad_token_id=0, teacher_top_k=TEACHER_TOP_K, num_processes=1, teacher_id_to_idx={"math": 0, "code": 1}
        )
        batch = collator([[[_rollout_sample(3, teacher_id="math"), _rollout_sample(2, teacher_id="code")]]])

        assert batch["teacher_id_idx"].tolist() == [[0, 0, 0, 1, 1]]

        single = DataCollatorForRollout(pad_token_id=0, teacher_top_k=TEACHER_TOP_K, num_processes=1)
        assert "teacher_id_idx" not in single([[[_rollout_sample(3)]]])


class TestMetricReduction:
    """The value's shape and the key's suffix are the whole reduction API; nothing is registered or configured."""

    @pytest.mark.parametrize(
        ("key", "values", "expected"),
        [
            ("perf/fwd_bwd_s", [1.0, 2.0, 6.0], 3.0),  # gauge -> window mean
            ("sample/dropped_stale_total", [1.0, 1.0, 1.0], 3.0),  # counter -> sum of the pushed deltas
            ("sample/forwarded_tokens_max", [3.0, 9.0, 5.0], 9.0),
            # A `max`/`min` WORD, not a suffix, so the upstream spellings reduce correctly too...
            ("completions/max_length", [3.0, 9.0, 5.0], 9.0),
            ("completions/min_length", [3.0, 9.0, 5.0], 3.0),
            # ... while a name that merely contains those letters stays a gauge.
            ("perf/maximal_s", [2.0, 4.0], 3.0),
            # Rates reduce as Σnum / Σden. These two windows ran at 1 and 100 tok/s; meaning the ratios would report
            # 50.5, the defect that made `training_tok/s` unusable.
            ("perf/forwarded_tok_s_wall_clock", [(100.0, 100.0), (100.0, 1.0)], 200 / 101),
        ],
    )
    def test_reduces_by_shape_and_name(self, key, values, expected):
        assert _reduce_metric(key, values) == pytest.approx(expected)

    def test_rate_with_zero_denominator_is_nan(self):
        assert math.isnan(_reduce_metric("perf/forwarded_tok_s_wall_clock", [(0.0, 0.0)]))


class TestRolloutWorkerProtocol:
    def test_stub_worker_exposes_every_protocol_attribute(self):
        # A stub that falls behind the protocol breaks training, and only the GPU-gated trainer test would notice.
        stub = _StubRolloutWorker(tokenizer=None, dataset=[], vocab_size=8)
        assert not [name for name in RolloutWorkerProtocol.__annotations__ if not hasattr(stub, name)]


def _position(*entries):
    """A vLLM `prompt_logprobs[i]` entry: `{token_id: {"logprob": ..., "rank": ...}}`, JSON keys are strings."""
    return {str(token_id): {"logprob": logprob, "rank": rank} for token_id, logprob, rank in entries}


class TestParseTeacherLogprobs:
    """vLLM's `prompt_logprobs` mapping is unordered; parsing sorts it by rank and drops NaNs."""

    @pytest.mark.parametrize(
        ("position", "expected_ids", "expected_logprobs"),
        [
            (_position((9, -2.0, 3), (5, -0.1, 1), (1, -1.0, 2)), [5, 1, 9], [-0.1, -1.0, -2.0]),
            # The realized token is reported whatever its rank, on top of the requested top-k. Truncating back to k
            # (as `VLLMClient.get_sequence_logprobs` does for the sync trainers) would drop it here, zeroing the
            # teacher signal at that position for beta=1.0.
            (_position((5, -0.1, 1), (1, -1.0, 2), (9, -8.0, 4242)), [5, 1, 9], [-0.1, -1.0, -8.0]),
            # NaN reflects an invalid score, not a padding slot, so the candidate is dropped rather than sentinelled.
            (_position((5, -0.1, 1), (1, float("nan"), 2), (9, -2.0, 3)), [5, 9], [-0.1, -2.0]),
            # A position vLLM scored nothing for yields the same empty pair as a masked-out prompt position.
            (None, [], []),
            ({}, [], []),
        ],
    )
    def test_parses_position(self, position, expected_ids, expected_logprobs):
        ids, logprobs = _parse_teacher_logprobs_at_position(position)
        assert ids == expected_ids
        assert logprobs == pytest.approx(expected_logprobs)


def _bare_loop(tokenizer, teacher_server_urls):
    # _generate_and_score_one/_score_with_teacher only read the attributes set below off self, so we skip the heavy
    # __init__ (mp.Queue/mp.Value/mp.Event, child-process bookkeeping) and set just those, mirroring
    # AsyncGRPOTrainer's own `_bare_loop` test helper.
    loop = object.__new__(_AsyncRolloutLoop)
    loop.tokenizer = tokenizer
    loop.chat_template_kwargs = {}
    loop.teacher_server_urls = teacher_server_urls
    # Normally resolved from each teacher's /v1/models in _run_loops, which no bare loop ever reaches.
    loop.teacher_model_names = {teacher_id: f"{teacher_id}-model" for teacher_id in teacher_server_urls}
    loop.teacher_top_k = TEACHER_TOP_K
    loop.teacher_temperature = 1.0
    loop.request_timeout = 30
    loop._model_version_value = types.SimpleNamespace(value=0)
    # `_generate_and_score_one` pushes its rollout metrics; collect them instead of sending them to a queue.
    loop._pushed_metrics = []
    loop._counters = defaultdict(float)
    loop._rates = defaultdict(lambda: [0.0, 0.0])
    loop._push_metrics = loop._pushed_metrics.append
    return loop


ONE_TEACHER = {"default": "http://default:8001"}
TWO_TEACHERS = {"math": "http://math:8002", "code": "http://code:8003"}


class TestWorkerMetrics:
    """The worker's payload has the same shape as the trainer's sink, so draining it in `log()` is an append."""

    def _loop(self, teacher_server_urls=ONE_TEACHER, maxsize=0):
        loop = object.__new__(_AsyncRolloutLoop)
        loop._metrics_queue = mp.Queue(maxsize=maxsize)
        loop._counters = defaultdict(float)
        loop._rates = defaultdict(lambda: [0.0, 0.0])
        loop.teacher_server_urls = teacher_server_urls
        loop.tokenizer = types.SimpleNamespace(eos_token_id=0, pad_token_id=0)
        return loop

    def test_counters_and_rates_ride_along_and_reset(self):
        loop = self._loop()
        loop._counters["rollout/vllm_retry_total"] += 2
        loop._rates["rollout/score_s"][0] += 3.0
        loop._rates["rollout/score_s"][1] += 2
        loop._push_metrics({"rollout/inflight": 4.0})

        assert loop._metrics_queue.get(timeout=5) == {
            "rollout/inflight": 4.0,
            "rollout/vllm_retry_total": 2.0,
            "rollout/score_s": (3.0, 2.0),
        }
        # Counters and rates carry deltas, so what one push reported must not ride along on the next.
        loop._push_metrics({})
        assert loop._metrics_queue.get(timeout=5) == {}

    def test_push_never_blocks_when_the_trainer_stops_draining(self):
        loop = self._loop(maxsize=1)
        for _ in range(50):
            loop._push_metrics({"rollout/inflight": 1.0})  # drops rather than stalling generation

    @pytest.mark.parametrize(
        ("teacher_server_urls", "completion_ids", "clipped", "per_teacher"),
        [
            (ONE_TEACHER, [7, 8, 0], 0.0, False),  # ends on eos: the model stopped on its own
            (TWO_TEACHERS, [7, 8, 9], 1.0, True),  # ends mid-sentence: cut off by max_completion_length
        ],
    )
    def test_rollout_push_reports_the_completion_and_the_teacher_call(
        self, teacher_server_urls, completion_ids, clipped, per_teacher
    ):
        loop = self._loop(teacher_server_urls)
        loop._push_rollout_metrics(completion_ids=completion_ids, teacher_id="math", score_s=0.25, duration_s=1.5)
        payload = loop._metrics_queue.get(timeout=5)

        assert payload["rollout/duration_s"] == 1.5
        assert payload["completions/mean_length"] == (3.0, 1.0)
        assert payload["completions/max_length"] == 3.0
        assert payload["completions/clipped_ratio"] == (clipped, 1.0)
        assert payload["rollout/score_s"] == (0.25, 1.0)
        # A per-teacher latency series only exists under MOPD; with one teacher the blended mean is the whole story.
        assert ("teacher_score_s/math" in payload) is per_teacher


class TestMultiTeacherRouting:
    """A single configured teacher scores every row regardless of `teacher_id` (plain on-policy distillation).

    Multiple teachers enable MOPD: each row's `teacher_id` column selects among them; a missing or unmapped
    `teacher_id` is a configuration error, not a silent fallback. Each request must carry the URL *and* the served
    model id of the teacher it routes to, since `/v1/completions` names the model it addresses and every teacher serves
    its own.
    """

    def _run(self, teacher_server_urls, *teacher_ids):
        """Score one row per `teacher_ids` entry (`None` = no `teacher_id` column) and return the requests sent."""
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        loop = _bare_loop(tokenizer, teacher_server_urls)
        requests = []

        async def fake_generate_one_turn(prompt_ids):
            return [7]

        async def fake_post(base_url, path, payload, timeout):
            requests.append((base_url, payload))
            # vLLM scores every token of the sequence it was handed, not just the completion region.
            return {"choices": [{"prompt_logprobs": [_position((7, -0.5, 1))] * len(payload["prompt"])}]}

        loop._generate_one_turn = fake_generate_one_turn
        loop._post = fake_post

        rows = [
            {"prompt": [{"role": "user", "content": "hi"}], **({} if tid is None else {"teacher_id": tid})}
            for tid in teacher_ids
        ]
        samples = [asyncio.run(loop._generate_and_score_one(row)) for row in rows]
        return samples, requests

    @pytest.mark.parametrize("teacher_id", ["whatever", None])
    def test_single_teacher_scores_every_row(self, teacher_id):
        samples, requests = self._run(ONE_TEACHER, teacher_id)
        assert [url for url, _ in requests] == ["http://default:8001"]
        assert [payload["model"] for _, payload in requests] == ["default-model"]
        assert [sample.teacher_id for sample in samples] == ["default"]

    def test_each_row_is_addressed_to_its_own_teacher(self):
        samples, requests = self._run(TWO_TEACHERS, "code", "math")
        assert [url for url, _ in requests] == ["http://code:8003", "http://math:8002"]
        assert [payload["model"] for _, payload in requests] == ["code-model", "math-model"]
        # The routed id rides along on the sample so the collator can tag every token with it (MOPD metrics).
        assert [sample.teacher_id for sample in samples] == ["code", "math"]

    def test_scores_only_the_completion_region(self):
        (sample,), ((_, payload),) = self._run(ONE_TEACHER, None)
        n_prompt = len(sample.input_ids) - sum(sample.completion_mask)
        assert payload["prompt"] == sample.input_ids
        assert payload["prompt_logprobs"] == TEACHER_TOP_K
        assert sample.teacher_topk_ids[:n_prompt] == [[]] * n_prompt
        assert sample.teacher_topk_ids[n_prompt:] == [[7]] * sum(sample.completion_mask)

    @pytest.mark.parametrize(("teacher_id", "match"), [(None, "teacher_id=None"), ("unknown", "teacher_id='unknown'")])
    def test_multi_teacher_raises_on_unroutable_row(self, teacher_id, match):
        with pytest.raises(ValueError, match=match):
            self._run(TWO_TEACHERS, teacher_id)


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
            rollout_worker=_StubRolloutWorker(tokenizer, dataset, vocab_size=len(tokenizer)),
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
            rollout_worker=_StubRolloutWorker(tokenizer, dataset, vocab_size=len(tokenizer)),
            weight_transfer=_StubWeightTransfer(),
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

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
            rollout_worker=_StubRolloutWorker(tokenizer, dataset, vocab_size=len(tokenizer)),
            weight_transfer=_StubWeightTransfer(),
        )
        trainer.train()

        log_history = [rec for rec in trainer.state.log_history if "grad_norm" in rec]
        assert log_history, "Expected at least one grad_norm log entry during training"
        for record in log_history:
            assert math.isfinite(record["grad_norm"]), f"grad_norm={record['grad_norm']} leaked -inf into backward"
            assert math.isfinite(record["loss"])
