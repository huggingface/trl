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

import copy
import importlib.util

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from trl.experimental.async_distillation.async_distillation_trainer import DataCollatorForRollout, _chunked_csd_loss
from trl.experimental.async_distillation.csd import CSDTargetBuilder, csd_divergence


# Optional upstream oracle: PYTHONPATH=../RL python -m pytest ... -k nemo
# Production TRL never imports or depends on NeMo.
if importlib.util.find_spec("nemo_rl") is not None:
    from nemo_rl.algorithms.loss.csd import CSDLossFn
    from nemo_rl.algorithms.x_token.loss_utils import LocalizedAlignment
else:
    CSDLossFn = None


def make_tokenizer(teacher=False, permute_ids=False):
    alphabet = sorted(ByteLevelPreTokenizer.alphabet())
    merges = (
        [("b", "c"), ("a", "bc"), ("x", "y"), ("b", "d")]
        if teacher
        else [("a", "b"), ("ab", "c"), ("x", "y"), ("a", "c")]
    )
    if permute_ids:
        alphabet = alphabet[::-1]
    tokens = alphabet + [a + b for a, b in merges]
    backend = Tokenizer(BPE(vocab={token: i for i, token in enumerate(tokens)}, merges=merges))
    backend.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False, use_regex=False)
    backend.decoder = ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>", pad_token="<eos>")


@pytest.fixture(scope="module")
def builder():
    return CSDTargetBuilder(make_tokenizer(), make_tokenizer(teacher=True), top_k=4)


def test_prefix_alternatives(builder):
    s = builder.student_tokenizer.convert_tokens_to_ids(["ab"])
    t = builder.teacher_tokenizer.convert_tokens_to_ids(["a", "b"])
    c = builder.teacher_tokenizer.convert_tokens_to_ids("c")
    ids, lps, _, _ = builder.build(s, t, builder.align(s, t), [[t[0]], [t[1], c]], [[-1.0], [-1.0, -2.0]], 5.0)
    assert ids[0, :2].tolist() == [s[0], builder.student_tokenizer.convert_tokens_to_ids("ac")]
    np.testing.assert_allclose(lps[0, :2], [-2.0, -3.0])


def test_noncanonical_generation_and_unicode(builder):
    s = builder.student_tokenizer.convert_tokens_to_ids(["a", "b", "c"])
    t = builder.teacher_tokenizer.encode("abc", add_special_tokens=False)
    assert s != builder.student_tokenizer.encode("abc", add_special_tokens=False)
    assert builder.align(s, t) == [(0, 3, 0, 1)]
    text = "café 🙂 中文"
    s = builder.student_tokenizer.encode(text, add_special_tokens=False)
    t = builder.teacher_tokenizer.encode(text, add_special_tokens=False)
    chunks = builder.align(s, t)
    assert sum(s1 - s0 for s0, s1, _, _ in chunks) == len(s)
    assert sum(t1 - t0 for _, _, t0, t1 in chunks) == len(t)
    assert builder.align(s, builder.teacher_tokenizer.encode("different", add_special_tokens=False)) == []


@pytest.fixture(scope="module")
def tiny_model():
    return (
        AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", dtype=torch.float32)
        .cpu()
        .eval()
    )


@pytest.fixture(scope="module")
def mixed_teachers(tiny_model):
    # Both experts differ from the student. The second reverses the byte ID
    # order as well, so using teacher IDs to gather student logits is wrong
    # even for ordinary single-byte tokens. These are actual BPE tokenizers.
    student = make_tokenizer()
    teachers = [make_tokenizer(teacher=True), make_tokenizer(teacher=True, permute_ids=True)]
    builders = [CSDTargetBuilder(student, teacher, top_k=4) for teacher in teachers]
    models = [copy.deepcopy(tiny_model) for _ in teachers]
    for model, tokenizer in zip(models, teachers, strict=True):
        model.resize_token_embeddings(len(tokenizer))
        model.requires_grad_(False)
    # Same coverage as NeMo's common/mismatch fixtures, with genuine token
    # events instead of pseudo tables. Include unequal teacher cohorts and a
    # sample with no valid teacher scores.
    cases = [
        (0, ["abc"], ["abc"]),
        (1, ["ab"], ["a", "b"]),
        (0, ["b", "c"], ["bc"]),
        (1, ["ab", "c"], ["a", "bc"]),
        (1, ["abc"], ["abc"]),
        (0, ["ab"], ["a", "b"]),
    ]
    samples = []
    for index, (expert, s_tokens, t_tokens) in enumerate(cases):
        teacher = teachers[expert]
        s_ids = student.convert_tokens_to_ids(s_tokens)
        t_ids = teacher.convert_tokens_to_ids(t_tokens)
        with torch.no_grad():
            teacher_logits = models[expert](torch.tensor([[teacher.eos_token_id] + t_ids])).logits[0]
            logps = teacher_logits[:-1].log_softmax(-1)
        # A full teacher row makes the support/mapping oracle independent of
        # random top-k membership, as in NeMo's full-support transport tests.
        top_ids = [list(range(len(teacher)))] * len(t_ids)
        top_lps = logps.tolist()
        if index == len(cases) - 1:
            top_ids[0], top_lps[0] = [], []
        chunks = builders[expert].align(s_ids, t_ids)
        ids, lps, lengths, weights = builders[expert].build(s_ids, t_ids, chunks, top_ids, top_lps, 5.0)
        samples.append(
            dict(
                expert=expert,
                s_ids=s_ids,
                t_ids=t_ids,
                chunks=chunks,
                teacher_logps=logps,
                teacher_logits=teacher_logits,
                input_ids=np.array([student.eos_token_id] + s_ids),
                completion_mask=np.array([0] + [1] * len(s_ids)),
                teacher_topk_ids=np.concatenate([np.full((1, 5), -1), ids]),
                teacher_topk_logprobs=np.concatenate([np.full((1, 5), -np.inf), lps]),
                csd_prefix_lengths=np.concatenate([[0], lengths]),
                csd_weights=np.concatenate([[0.0], weights]).astype(np.float32),
                teacher_id=str(expert),
                prompt_id=index,
                metrics={},
            )
        )
    return student, teachers, samples


def probability_divergence(q, p, beta):
    # Independent probability-space reference, matching NeMo's parity tests.
    if beta == 0:
        return (p * (p.log() - q.log())).sum()
    if beta == 1:
        return (q * (q.log() - p.log())).sum()
    mixture = (1 - beta) * q + beta * p
    return (beta * p * (p.log() - mixture.log()) + (1 - beta) * q * (q.log() - mixture.log())).sum()


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("chunk_size", [1, 3])
def test_mixed_tokenizers_packed_loss_and_gradients_match_dense_oracle(tiny_model, mixed_teachers, beta, chunk_size):
    student, teachers, samples = mixed_teachers
    model = copy.deepcopy(tiny_model)
    model.resize_token_embeddings(len(student))
    collator = DataCollatorForRollout(student.pad_token_id, 2, {"0": 0, "1": 1})
    # Unequal packed rows; teacher 1 contributes three chunks while teacher 0
    # contributes two. The final sample has no usable teacher-prefix score.
    batch = collator([[samples[:2], samples[2:]]])
    hidden, dense_logps = [], []
    for sample in samples:
        tokens = torch.as_tensor(sample["input_ids"]).unsqueeze(0)
        hidden.append(model.base_model(input_ids=tokens).last_hidden_state[0, :-1])
        dense_logps.append(model(tokens).logits[0, :-1].double().log_softmax(-1))
    targets = torch.cat([torch.as_tensor(sample["teacher_topk_ids"][1:]) for sample in samples])
    teacher_events = torch.cat([torch.as_tensor(sample["teacher_topk_logprobs"][1:]) for sample in samples]).float()
    prefixes = torch.cat([torch.as_tensor(sample["csd_prefix_lengths"][1:]) for sample in samples])
    weights = torch.cat([torch.as_tensor(sample["csd_weights"][1:]) for sample in samples])
    actual_ids = torch.tensor([token for sample in samples for token in sample["s_ids"]])
    expert_ids = torch.tensor([sample["expert"] for sample in samples for _ in sample["s_ids"]])
    result = _chunked_csd_loss(
        torch.cat(hidden),
        model.get_output_embeddings().weight,
        None,
        1.0,
        None,
        targets,
        teacher_events,
        actual_ids,
        prefixes,
        weights,
        beta,
        1.0,
        expert_ids,
        2,
        chunk_size,
    )
    actual_loss = result[0] / batch["global_n_chunks"][0]
    actual_loss.backward()
    actual_grads = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}
    model.zero_grad()
    reference = dense_logps[0].new_zeros(())
    count = 0
    for sample, student_logps in zip(samples, dense_logps, strict=True):
        teacher = teachers[sample["expert"]]
        for s0, s1, t0, t1 in sample["chunks"]:
            if sample["csd_weights"][s1] == 0:
                continue
            q_events, p_events = [], []
            s_prefix = sample["s_ids"][s0 : s1 - 1]
            t_prefix = sample["t_ids"][t0 : t1 - 1]
            for sid in sample["teacher_topk_ids"][s1]:
                if sid < 0:
                    continue
                event = "".join(student.convert_ids_to_tokens(s_prefix + [int(sid)]))
                # Brute-force vocabulary surfaces form an independent mapping
                # oracle: never read the production sub-token tables here.
                matching = [
                    tid
                    for tid in range(len(teacher))
                    if "".join(teacher.convert_ids_to_tokens(t_prefix + [tid])) == event
                ]
                assert len(matching) == 1
                tid = matching[0]
                s_lp = sum(student_logps[pos, token] for pos, token in enumerate(s_prefix, start=s0))
                t_lp = sum(
                    sample["teacher_logps"][pos, token].double() for pos, token in enumerate(t_prefix, start=t0)
                )
                q_events.append((s_lp + student_logps[s1 - 1, sid]).exp())
                p_events.append((t_lp + sample["teacher_logps"][t1 - 1, tid].double()).exp())
            q, p = torch.stack(q_events), torch.stack(p_events)
            q = torch.cat([q, (1 - q.sum()).unsqueeze(0)])
            p = torch.cat([p, (1 - p.sum()).unsqueeze(0)])
            reference = reference + sample["csd_weights"][s1] * probability_divergence(q, p, beta)
            count += 1
    reference = reference / count
    reference.backward()
    assert count == 5
    torch.testing.assert_close(actual_loss.double(), reference, atol=2e-7, rtol=1e-4)
    for name, param in model.named_parameters():
        if name in actual_grads:
            assert torch.isfinite(actual_grads[name]).all()
            torch.testing.assert_close(actual_grads[name], param.grad, atol=2e-6, rtol=3e-4)
    assert result[4][::3].tolist() == [2.0, 3.0]


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_softmax_cancellation_at_and_near_matched_cross_tokenizer_events(beta):
    # Unequal chain event masses plus REST, including a nearly deterministic
    # event. The teacher support is already mapped across tokenizer IDs.
    teacher = torch.tensor([[-0.4, -2.0, -4.0], [-2e-8, -20.0, -22.0]], dtype=torch.float32)
    mask = torch.ones_like(teacher, dtype=torch.bool)
    equal = teacher.clone().requires_grad_()
    loss, _ = csd_divergence(equal, teacher, mask, beta)
    loss.sum().backward()
    assert torch.equal(loss, torch.zeros_like(loss))
    assert torch.equal(equal.grad, torch.zeros_like(equal))
    near = (teacher + torch.tensor([[2e-7, -2e-7, 1e-7], [-1e-9, 2e-6, -2e-6]])).requires_grad_()
    actual, _ = csd_divergence(near, teacher, mask, beta)
    actual.sum().backward()
    # Double-precision probability equations retain the true tiny signal and
    # provide an independent gradient oracle for Barron's cancellation case.
    oracle = near.detach().double().requires_grad_()
    expected = []
    for s, t in zip(oracle, teacher.double(), strict=True):
        q, p = s.exp(), t.exp()
        q = torch.cat([q, (1 - q.sum()).unsqueeze(0)])
        p = torch.cat([p, (1 - p.sum()).unsqueeze(0)])
        expected.append(probability_divergence(q, p, beta))
    torch.stack(expected).sum().backward()
    assert torch.isfinite(near.grad).all()
    assert near.grad.abs().sum() > 0
    torch.testing.assert_close(near.grad.double(), oracle.grad, atol=2e-13, rtol=3e-5)


@pytest.mark.skipif(CSDLossFn is None, reason="requires NVIDIA-NeMo/RL PR #4680 on PYTHONPATH")
@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_mixed_tokenizers_match_official_nemo_loss_and_gradients(tmp_path, tiny_model, mixed_teachers, beta):
    student, teachers, samples = mixed_teachers
    model = copy.deepcopy(tiny_model)
    model.resize_token_embeddings(len(student))
    n_chunks = sum(int((sample["csd_weights"] > 0).sum()) for sample in samples)
    ours, official = [], []
    for expert, teacher in enumerate(teachers):
        builder = CSDTargetBuilder(student, teacher, top_k=len(teacher))
        forward_path, reverse_path = tmp_path / f"forward{expert}.pt", tmp_path / f"reverse{expert}.pt"
        torch.save({"subtoks": builder.forward, "lengths": builder.forward_lengths}, forward_path)
        torch.save({"subtoks": builder.reverse, "lengths": builder.reverse_lengths}, reverse_path)
        upstream = CSDLossFn(
            dict(
                student_vocab_size=len(student),
                teacher_vocab_sizes=[len(teacher)],
                student_to_teacher_table_paths=[str(forward_path)],
                teacher_to_student_table_paths=[str(reverse_path)],
                common_indices_from_subtoks=True,
                teacher_topk_ipc_k=0,
                teacher_topk_ipc_keep_realized=True,
                csd_loss_fn="jsd",
                csd_jsd_beta=beta,
                csd_mismatch_loss_beta=5.0,
            )
        )
        for sample in samples:
            if sample["expert"] != expert or not (sample["csd_weights"] > 0).any():
                continue
            s_ids, t_ids = sample["s_ids"], sample["t_ids"]
            t_logps = sample["teacher_logps"]
            ids, lps, lengths, weights = builder.build(
                s_ids, t_ids, sample["chunks"], [list(range(len(teacher)))] * len(t_ids), t_logps.tolist(), 5.0
            )
            tokens = torch.as_tensor(sample["input_ids"]).unsqueeze(0)
            hidden = model.base_model(input_ids=tokens).last_hidden_state[0, :-1]
            loss, *_ = _chunked_csd_loss(
                hidden,
                model.get_output_embeddings().weight,
                None,
                1.0,
                None,
                torch.as_tensor(ids),
                torch.as_tensor(lps),
                torch.tensor(s_ids),
                torch.as_tensor(lengths),
                torch.as_tensor(weights),
                beta,
                1.0,
                None,
                1,
                2,
            )
            ours.append(loss / n_chunks)
            spans = sample["chunks"]
            alignment = LocalizedAlignment(
                sample_mask=torch.ones(1, dtype=torch.bool),
                student_input_ids=tokens,
                teacher_input_ids=torch.tensor([[teacher.eos_token_id] + t_ids]),
                student_spans=torch.tensor([[[a + 1, b + 1] for a, b, _, _ in spans]]),
                teacher_spans=torch.tensor([[[a + 1, b + 1] for _, _, a, b in spans]]),
                pair_valid=torch.ones((1, len(spans)), dtype=torch.bool),
                num_chunks=torch.tensor([len(spans)]),
            )
            reference, _ = upstream._compute_csd_loss(
                0,
                model(tokens).logits,
                sample["teacher_logits"].unsqueeze(0),
                alignment,
                teacher_vocab_size=len(teacher),
                global_valid_chunks=torch.tensor(float(n_chunks)),
            )
            official.append(reference)
    actual_loss, reference_loss = torch.stack(ours).sum(), torch.stack(official).sum()
    actual_loss.backward()
    actual_grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()
    reference_loss.backward()
    torch.testing.assert_close(actual_loss, reference_loss, atol=3e-6, rtol=2e-4)
    for name, p in model.named_parameters():
        if name in actual_grads:
            torch.testing.assert_close(actual_grads[name], p.grad, atol=3e-6, rtol=5e-4)
