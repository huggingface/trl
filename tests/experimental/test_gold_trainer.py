# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

from types import SimpleNamespace

import pytest
import torch
from transformers import AutoTokenizer

from trl.experimental.gold.gold_trainer import GOLDTrainer, ULDLoss


def build_config(**overrides):
    base = dict(
        uld_crossentropy_weight=0.0,
        uld_distillation_weight=1.0,
        uld_student_temperature=1.0,
        uld_teacher_temperature=1.0,
        uld_skip_student_eos=False,
        uld_skip_teacher_eos=False,
        use_extended_uld=True,
        uld_use_hybrid_loss=False,
        uld_hybrid_matched_weight=None,
        uld_hybrid_unmatched_weight=None,
        beta=0.5,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture(scope="session")
def llama_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture(scope="session")
def qwen_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture(scope="session")
def smollm_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def encode_prompt_completion(tokenizer, prompt, completion):
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        completion_ids = completion_ids + [eos_id]
    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    return input_ids, labels


def pad_tokens(ids, pad_id, target_length):
    return ids + [pad_id] * (target_length - len(ids))


def pad_labels(labels, target_length):
    return labels + [-100] * (target_length - len(labels))


def test_alignment_groups_cover_all_tokens(llama_tokenizer, qwen_tokenizer):
    config = build_config()
    loss = ULDLoss(config, student_tokenizer=llama_tokenizer, teacher_tokenizer=qwen_tokenizer)

    text = "SmolLM3-3B is smaller than Llama 3.2 but still capable."
    student_ids = llama_tokenizer(text, add_special_tokens=False)["input_ids"]
    teacher_ids = qwen_tokenizer(text, add_special_tokens=False)["input_ids"]

    student_groups, teacher_groups = loss._build_alignment_groups_from_ids(student_ids, teacher_ids)

    assert len(student_groups) == len(teacher_groups)
    assert sorted(idx for group in student_groups for idx in group) == list(range(len(student_ids)))
    assert sorted(idx for group in teacher_groups for idx in group) == list(range(len(teacher_ids)))


def test_merge_probabilities_multiplies_split_tokens():
    config = build_config()
    # Use simple 3-token vocabulary to validate merging behaviour
    probs = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
    loss = ULDLoss(config, student_tokenizer=None, teacher_tokenizer=None)

    merged = loss._merge_probabilities_with_alignment_groups(probs, [[0, 1]])
    expected = torch.softmax(torch.log(probs[0]) + torch.log(probs[1]), dim=-1)

    assert torch.allclose(merged[0], expected, atol=1e-6)


def test_initialize_vocabulary_mapping_contains_common_tokens(llama_tokenizer, qwen_tokenizer):
    config = build_config(
        uld_use_hybrid_loss=True,
        uld_hybrid_matched_weight=1.0,
        uld_hybrid_unmatched_weight=0.0,
    )
    loss = ULDLoss(config, student_tokenizer=llama_tokenizer, teacher_tokenizer=qwen_tokenizer)

    common_tokens = ["Hello", "Ġworld", "-", "Sm", "ol", "LM", "3", "B"]
    for token in common_tokens:
        student_id = llama_tokenizer.convert_tokens_to_ids(token)
        teacher_id = qwen_tokenizer.convert_tokens_to_ids(token)
        assert student_id is not None
        assert teacher_id is not None
        assert teacher_id in loss._vocab_mapping
        assert loss._vocab_mapping[teacher_id] == student_id
        assert teacher_id in loss._teacher_matched_ids
        assert student_id in loss._student_matched_ids


def test_generalized_jsd_loss_accepts_probability_inputs():
    student_probs = torch.tensor([[[0.6, 0.3, 0.1]]])
    teacher_probs = torch.tensor([[[0.5, 0.4, 0.1]]])
    mixture = 0.5 * (student_probs + teacher_probs)
    expected = 0.5 * (
        torch.sum(student_probs * (torch.log(student_probs) - torch.log(mixture)))
        + torch.sum(teacher_probs * (torch.log(teacher_probs) - torch.log(mixture)))
    )

    loss = GOLDTrainer.generalized_jsd_loss(
        student_probs,
        teacher_probs,
        beta=0.5,
        reduction="batchmean",
        logits_are_probs=True,
    )

    assert torch.allclose(loss, expected, atol=1e-6)


def test_uldloss_handles_llama_student_qwen_teacher_sequence(llama_tokenizer, qwen_tokenizer):
    config = build_config(
        uld_use_hybrid_loss=True,
        uld_hybrid_matched_weight=0.6,
        uld_hybrid_unmatched_weight=0.4,
    )
    loss_fn = ULDLoss(config, student_tokenizer=llama_tokenizer, teacher_tokenizer=qwen_tokenizer)

    prompt = "User: Summarize the difference between llamas and alpacas."
    completion = "Assistant: Llamas are taller while alpacas have softer wool."

    student_ids, student_labels = encode_prompt_completion(llama_tokenizer, prompt, completion)
    teacher_ids, teacher_labels = encode_prompt_completion(qwen_tokenizer, prompt, completion)

    pad_id_student = llama_tokenizer.pad_token_id
    pad_id_teacher = qwen_tokenizer.pad_token_id
    max_length = max(len(student_ids), len(teacher_ids))

    student_ids = pad_tokens(student_ids, pad_id_student, max_length)
    teacher_ids = pad_tokens(teacher_ids, pad_id_teacher, max_length)
    student_labels = pad_labels(student_labels, max_length)
    teacher_labels = pad_labels(teacher_labels, max_length)

    student_input_ids = torch.tensor([student_ids])
    teacher_input_ids = torch.tensor([teacher_ids])
    student_labels = torch.tensor([student_labels])
    teacher_labels = torch.tensor([teacher_labels])

    student_vocab = len(llama_tokenizer)
    teacher_vocab = len(qwen_tokenizer)

    student_logits = torch.randn(1, max_length, student_vocab)
    teacher_logits = torch.randn(1, max_length, teacher_vocab)

    loss = loss_fn(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        student_labels=student_labels,
        teacher_labels=teacher_labels,
        student_input_ids=student_input_ids,
        teacher_input_ids=teacher_input_ids,
    )

    assert torch.isfinite(loss)
    assert loss.dim() == 0
    assert loss_fn.last_matched_loss is not None
    assert loss_fn.last_unmatched_loss is not None


def test_uldloss_handles_smollm_student_qwen_teacher_sequence(smollm_tokenizer, qwen_tokenizer):
    config = build_config(
        uld_use_hybrid_loss=True,
        uld_hybrid_matched_weight=0.5,
        uld_hybrid_unmatched_weight=0.5,
    )
    loss_fn = ULDLoss(config, student_tokenizer=smollm_tokenizer, teacher_tokenizer=qwen_tokenizer)

    prompt = "User: Describe SmolLM3 in a sentence."
    completion = "Assistant: SmolLM3 is a compact yet capable language model."

    student_ids, student_labels = encode_prompt_completion(smollm_tokenizer, prompt, completion)
    teacher_ids, teacher_labels = encode_prompt_completion(qwen_tokenizer, prompt, completion)

    pad_id_student = smollm_tokenizer.pad_token_id
    pad_id_teacher = qwen_tokenizer.pad_token_id
    max_length = max(len(student_ids), len(teacher_ids))

    student_ids = pad_tokens(student_ids, pad_id_student, max_length)
    teacher_ids = pad_tokens(teacher_ids, pad_id_teacher, max_length)
    student_labels = pad_labels(student_labels, max_length)
    teacher_labels = pad_labels(teacher_labels, max_length)

    student_input_ids = torch.tensor([student_ids])
    teacher_input_ids = torch.tensor([teacher_ids])
    student_labels = torch.tensor([student_labels])
    teacher_labels = torch.tensor([teacher_labels])

    student_vocab = len(smollm_tokenizer)
    teacher_vocab = len(qwen_tokenizer)

    student_logits = torch.randn(1, max_length, student_vocab)
    teacher_logits = torch.randn(1, max_length, teacher_vocab)

    loss = loss_fn(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        student_labels=student_labels,
        teacher_labels=teacher_labels,
        student_input_ids=student_input_ids,
        teacher_input_ids=teacher_input_ids,
    )

    assert torch.isfinite(loss)
    assert loss.dim() == 0
    assert loss_fn.last_matched_loss is not None
    assert loss_fn.last_unmatched_loss is not None
