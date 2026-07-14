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
from functools import partial
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset, load_dataset
from examples.scripts.gold import split_text_for_gold
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from trl.experimental.gold import GOLDConfig
from trl.experimental.gold import gold_trainer as gold_trainer_module
from trl.experimental.gold.gold_trainer import (
    GOLDTrainer,
    ULDLoss,
    build_teacher_inputs_from_texts,
)
from trl.experimental.utils import (
    DataCollatorForChatML,
    DataCollatorForVisionLanguageChatML,
    encode_with_byte_offsets,
    pad_byte_offsets,
)
from trl.trainer.utils import RepeatSampler, identity

from ..testing_utils import TrlTestCase, require_liger_kernel


@pytest.fixture(scope="module")
def openr1_examples():
    try:
        dataset = load_dataset(
            "HuggingFaceTB/OpenR1-Math-220k-default-verified",
            "all",
            split="train[:3]",
        )
    except Exception as exc:  # pragma: no cover - network/environment dependent
        pytest.skip(f"OpenR1 dataset unavailable: {exc}")
    return [{"messages": row["messages"]} for row in dataset]


@pytest.fixture(scope="module")
def countdown_examples():
    try:
        dataset = load_dataset(
            "HuggingFaceTB/Countdown-Tasks-3to4",
            "gkd_verified_Qwen2.5-7B-Instruct",
            split="train[:3]",
        )
    except Exception as exc:  # pragma: no cover - network/environment dependent
        pytest.skip(f"Countdown dataset unavailable: {exc}")
    return [{"messages": row["messages"]} for row in dataset]


def _teacher_inputs_from_collator(student_tok, teacher_tok, batch):
    prompt_texts = []
    completion_texts = []

    pad_token_id = student_tok.pad_token_id
    for prompt_ids_tensor, input_ids_tensor, labels_tensor in zip(
        batch["prompts"], batch["input_ids"], batch["labels"], strict=True
    ):
        prompt_ids = prompt_ids_tensor.tolist()
        if pad_token_id is not None:
            prompt_ids = [tok for tok in prompt_ids if tok != pad_token_id]
        prompt_texts.append(student_tok.decode(prompt_ids, skip_special_tokens=False))

        input_ids = input_ids_tensor.tolist()
        labels = labels_tensor.tolist()
        completion_token_ids = [tok for tok, label in zip(input_ids, labels, strict=True) if label != -100]
        completion_texts.append(student_tok.decode(completion_token_ids, skip_special_tokens=False))

    teacher_input_ids, teacher_labels, _, teacher_byte_offsets = build_teacher_inputs_from_texts(
        teacher_tok, prompt_texts, completion_texts
    )
    return teacher_input_ids, teacher_labels, completion_texts, teacher_byte_offsets


def _assert_alignment_covers_completion(loss_fn, batch, teacher_input_ids, teacher_labels, teacher_byte_offsets):
    """Assert byte-offset alignment groups cover every answer-region position."""
    for idx in range(batch["input_ids"].shape[0]):
        s_positions = (batch["labels"][idx] != -100).nonzero(as_tuple=True)[0]
        t_positions = (teacher_labels[idx] != -100).nonzero(as_tuple=True)[0]
        s_answer = batch["byte_offsets"][idx, s_positions[0] : s_positions[-1] + 1].tolist()
        t_answer = teacher_byte_offsets[idx, t_positions[0] : t_positions[-1] + 1].tolist()
        student_groups, teacher_groups = loss_fn._align_by_byte_offsets(s_answer, t_answer)
        assert student_groups and teacher_groups
        assert sorted(k for group in student_groups for k in group) == list(range(len(s_answer)))
        assert sorted(k for group in teacher_groups for k in group) == list(range(len(t_answer)))


@pytest.mark.slow
def test_chatml_collator_preserves_completion_llama(llama_tokenizer, qwen_tokenizer, openr1_examples):
    collator = DataCollatorForChatML(tokenizer=llama_tokenizer, max_length=512)
    batch = collator(openr1_examples)

    assistant_texts = [example["messages"][-1]["content"] for example in openr1_examples]
    decoded_batch = llama_tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=False)
    for decoded, assistant in zip(decoded_batch, assistant_texts, strict=True):
        assert assistant.strip() in decoded

    teacher_input_ids, teacher_labels, completion_texts, teacher_byte_offsets = _teacher_inputs_from_collator(
        llama_tokenizer, qwen_tokenizer, batch
    )
    for completion, assistant in zip(completion_texts, assistant_texts, strict=True):
        assert assistant.strip() in completion
        assert completion.strip()

    config = build_config(
        uld_use_hybrid_loss=True,
        uld_hybrid_matched_weight=0.6,
        uld_hybrid_unmatched_weight=0.4,
    )
    loss_fn = ULDLoss(config, student_tokenizer=llama_tokenizer, teacher_tokenizer=qwen_tokenizer)

    _assert_alignment_covers_completion(loss_fn, batch, teacher_input_ids, teacher_labels, teacher_byte_offsets)

    torch.manual_seed(0)
    student_vocab = len(llama_tokenizer)
    teacher_vocab = len(qwen_tokenizer)
    batch_size, seq_len = batch["input_ids"].shape
    student_logits = torch.randn(batch_size, seq_len, student_vocab)
    teacher_logits = torch.randn(batch_size, teacher_input_ids.shape[1], teacher_vocab)

    loss = loss_fn(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        student_labels=batch["labels"],
        teacher_labels=teacher_labels,
        student_input_ids=batch["input_ids"],
        teacher_input_ids=teacher_input_ids,
        student_byte_offsets=batch["byte_offsets"],
        teacher_byte_offsets=teacher_byte_offsets,
    )

    assert torch.isfinite(loss)


@pytest.mark.slow
def test_chatml_collator_preserves_completion_llama_countdown(llama_tokenizer, qwen_tokenizer, countdown_examples):
    collator = DataCollatorForChatML(tokenizer=llama_tokenizer, max_length=512)
    batch = collator(countdown_examples)

    assistant_texts = [example["messages"][-1]["content"] for example in countdown_examples]
    decoded_batch = llama_tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=False)
    for decoded, assistant in zip(decoded_batch, assistant_texts, strict=True):
        assert assistant.strip() in decoded

    teacher_input_ids, teacher_labels, completion_texts, teacher_byte_offsets = _teacher_inputs_from_collator(
        llama_tokenizer, qwen_tokenizer, batch
    )
    for completion, assistant in zip(completion_texts, assistant_texts, strict=True):
        assert assistant.strip() in completion
        assert completion.strip()

    config = build_config(
        uld_use_hybrid_loss=True,
        uld_hybrid_matched_weight=0.6,
        uld_hybrid_unmatched_weight=0.4,
    )
    loss_fn = ULDLoss(config, student_tokenizer=llama_tokenizer, teacher_tokenizer=qwen_tokenizer)

    _assert_alignment_covers_completion(loss_fn, batch, teacher_input_ids, teacher_labels, teacher_byte_offsets)

    torch.manual_seed(2)
    student_vocab = len(llama_tokenizer)
    teacher_vocab = len(qwen_tokenizer)
    batch_size, seq_len = batch["input_ids"].shape
    student_logits = torch.randn(batch_size, seq_len, student_vocab)
    teacher_logits = torch.randn(batch_size, teacher_input_ids.shape[1], teacher_vocab)

    loss = loss_fn(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        student_labels=batch["labels"],
        teacher_labels=teacher_labels,
        student_input_ids=batch["input_ids"],
        teacher_input_ids=teacher_input_ids,
        student_byte_offsets=batch["byte_offsets"],
        teacher_byte_offsets=teacher_byte_offsets,
    )

    assert torch.isfinite(loss)


@pytest.mark.slow
def test_chatml_collator_preserves_completion_smollm(smollm_tokenizer, qwen_tokenizer, openr1_examples):
    collator = DataCollatorForChatML(tokenizer=smollm_tokenizer, max_length=512)
    batch = collator(openr1_examples)

    assistant_texts = [example["messages"][-1]["content"] for example in openr1_examples]
    decoded_batch = smollm_tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=False)
    for decoded, assistant in zip(decoded_batch, assistant_texts, strict=True):
        assert assistant.strip() in decoded

    teacher_input_ids, teacher_labels, completion_texts, teacher_byte_offsets = _teacher_inputs_from_collator(
        smollm_tokenizer, qwen_tokenizer, batch
    )
    for completion, assistant in zip(completion_texts, assistant_texts, strict=True):
        assert assistant.strip() in completion
        assert completion.strip()

    config = build_config(
        uld_use_hybrid_loss=True,
        uld_hybrid_matched_weight=0.5,
        uld_hybrid_unmatched_weight=0.5,
    )
    loss_fn = ULDLoss(config, student_tokenizer=smollm_tokenizer, teacher_tokenizer=qwen_tokenizer)

    _assert_alignment_covers_completion(loss_fn, batch, teacher_input_ids, teacher_labels, teacher_byte_offsets)

    torch.manual_seed(1)
    student_vocab = len(smollm_tokenizer)
    teacher_vocab = len(qwen_tokenizer)
    batch_size, seq_len = batch["input_ids"].shape
    student_logits = torch.randn(batch_size, seq_len, student_vocab)
    teacher_logits = torch.randn(batch_size, teacher_input_ids.shape[1], teacher_vocab)

    loss = loss_fn(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        student_labels=batch["labels"],
        teacher_labels=teacher_labels,
        student_input_ids=batch["input_ids"],
        teacher_input_ids=teacher_input_ids,
        student_byte_offsets=batch["byte_offsets"],
        teacher_byte_offsets=teacher_byte_offsets,
    )

    assert torch.isfinite(loss)


def build_config(**overrides):
    base = dict(
        uld_crossentropy_weight=0.0,
        uld_distillation_weight=1.0,
        uld_student_temperature=1.0,
        uld_teacher_temperature=1.0,
        uld_skip_student_eos=False,
        uld_skip_teacher_eos=False,
        use_extended_uld=True,
        uld_token_merge_strategy="observed",
        uld_use_hybrid_loss=False,
        uld_hybrid_matched_weight=None,
        uld_hybrid_unmatched_weight=None,
        beta=0.5,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture(scope="session")
def llama_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-LlamaForCausalLM-3.2")
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


@pytest.fixture(scope="session")
def gemma4_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Gemma4ForConditionalGeneration")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture(scope="session")
def smolvlm_processor():
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return processor


@pytest.fixture(scope="session")
def qwen3_vl_processor():
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return processor


@pytest.fixture(scope="module")
def vlm_dataset():
    try:
        return load_dataset(
            "trl-internal-testing/zen-image",
            "conversational_prompt_completion",
            split="train[:3]",
        )
    except Exception as exc:  # pragma: no cover - network/environment dependent
        pytest.skip(f"zen-image dataset unavailable: {exc}")


@pytest.fixture
def vlm_examples(vlm_dataset):
    return [dict(row) for row in vlm_dataset]


def encode_prompt_completion(tokenizer, prompt, completion):
    """Build input_ids, labels, and per-token byte offsets for a (prompt, completion) pair.

    Byte offsets are computed via `encode_with_byte_offsets` on the completion text only, then padded with (0, 0) for
    prompt positions and a final (content_len, content_len) for the appended EOS — matching the shape produced by
    DataCollatorForChatML and build_teacher_inputs_from_texts.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    [(enc_ids, enc_offsets)] = encode_with_byte_offsets(
        tokenizer.backend_tokenizer, [completion], add_special_tokens=False
    )
    completion_ids = list(enc_ids)
    completion_offsets = list(enc_offsets)
    content_len = len(completion.encode("utf-8"))
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        completion_ids = completion_ids + [eos_id]
        completion_offsets = completion_offsets + [(content_len, content_len)]
    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    byte_offsets = [(0, 0)] * len(prompt_ids) + completion_offsets
    return input_ids, labels, byte_offsets


def pad_tokens(ids, pad_id, target_length):
    return ids + [pad_id] * (target_length - len(ids))


def pad_labels(labels, target_length):
    return labels + [-100] * (target_length - len(labels))


def test_process_completions_to_buffer_left_pads_prompt_ids():
    class RecordingTokenizer:
        pad_token_id = 0
        pad_token = "<pad>"

        def batch_decode(
            self,
            sequences,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ):
            del skip_special_tokens, clean_up_tokenization_spaces
            return [" ".join(str(token) for token in sequence) for sequence in sequences]

        def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
            del skip_special_tokens, clean_up_tokenization_spaces
            return " ".join(str(token) for token in ids)

    trainer = GOLDTrainer.__new__(GOLDTrainer)
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
    trainer.processing_class = RecordingTokenizer()
    trainer._tokenizer = RecordingTokenizer()
    trainer.args = SimpleNamespace(max_length=None)
    trainer.use_uld_loss = False
    trainer._buffered_inputs = [None]
    trainer._buffered_text_logs = [None]

    GOLDTrainer._process_completions_to_buffer(
        trainer,
        slices=[{"slice": "original"}],
        on_policy_indices=[0],
        local_slice_indices=[0, 0],
        completion_ids=[[31], [41]],
        prompt_ids_list=[[11], [21, 22]],
        prompts_text=["short", "longer"],
        max_completion_length=1,
    )

    buffered_inputs = trainer._buffered_inputs[0]
    assert torch.equal(
        buffered_inputs["input_ids"],
        torch.tensor([[0, 11, 31], [21, 22, 41]], dtype=torch.long),
    )
    assert torch.equal(
        buffered_inputs["attention_mask"],
        torch.tensor([[0, 1, 1], [1, 1, 1]], dtype=torch.long),
    )
    assert torch.equal(buffered_inputs["labels"], torch.tensor([[-100, -100, 31], [-100, -100, 41]]))


def test_generate_on_policy_for_slices_uses_prompt_attention_mask_for_vllm_prompts():
    class RecordingVLLMGeneration:
        def __init__(self):
            self.prompts = None
            self.sync_calls = 0

        def sync_weights(self):
            self.sync_calls += 1

        def generate(self, prompts, images, num_generations):
            self.prompts = prompts
            assert images is None
            assert num_generations == 1
            return None, [[42]], None, None

    class RecordingTokenizer:
        pad_token_id = 9
        pad_token = "<eos>"

        def batch_decode(
            self,
            sequences,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ):
            del clean_up_tokenization_spaces
            decoded = []
            token_map = {5: "A", 6: "B", 9: "<eos>"}
            for sequence in sequences:
                tokens = []
                for token in sequence:
                    token = int(token)
                    if skip_special_tokens and token == 9:
                        continue
                    tokens.append(token_map[token])
                decoded.append(" ".join(tokens))
            return decoded

        def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
            return self.batch_decode([ids], skip_special_tokens, clean_up_tokenization_spaces)[0]

    captured = {}

    def capture_process_completions(
        slices,
        on_policy_indices,
        local_slice_indices,
        completion_ids,
        prompt_ids_list,
        prompts_text,
        max_completion_length,
    ):
        captured["slices"] = slices
        captured["on_policy_indices"] = on_policy_indices
        captured["local_slice_indices"] = local_slice_indices
        captured["completion_ids"] = completion_ids
        captured["prompt_ids_list"] = prompt_ids_list
        captured["prompts_text"] = prompts_text
        captured["max_completion_length"] = max_completion_length

    trainer = GOLDTrainer.__new__(GOLDTrainer)
    trainer.accelerator = SimpleNamespace(is_main_process=True)
    trainer.args = SimpleNamespace(report_to=[])
    trainer.processing_class = RecordingTokenizer()
    trainer._tokenizer = RecordingTokenizer()
    trainer.use_vllm = True
    trainer.vllm_generation = RecordingVLLMGeneration()
    trainer.vllm_sync_frequency = 1
    trainer._last_vllm_sync_step = -1
    trainer.state = SimpleNamespace(global_step=0)
    trainer.num_generations = 1
    trainer.generation_config = SimpleNamespace(max_new_tokens=1)
    trainer._process_completions_to_buffer = capture_process_completions

    slices = [
        {
            "prompts": torch.tensor([[9, 9, 5, 9, 6]], dtype=torch.long),
            "prompt_attention_mask": torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.long),
        }
    ]

    GOLDTrainer._generate_on_policy_for_slices(trainer, slices, [0])

    assert trainer.vllm_generation.prompts == [[5, 9, 6]]
    assert trainer.vllm_generation.sync_calls == 1
    assert captured["completion_ids"] == [[42]]
    assert captured["prompt_ids_list"] == [[5, 9, 6]]
    assert captured["prompts_text"] == ["A <eos> B"]


def test_generate_on_policy_for_slices_reconstructs_prompt_with_special_tokens():
    class RecordingVLLMGeneration:
        def __init__(self):
            self.prompts = None
            self.sync_calls = 0

        def sync_weights(self):
            self.sync_calls += 1

        def generate(self, prompts, images, num_generations):
            self.prompts = prompts
            assert images is None
            assert num_generations == 1
            return None, [[42]], None, None

    class RecordingTokenizer:
        pad_token_id = 0
        pad_token = "<pad>"

        def __init__(self):
            self.truncation_side = "right"

        def batch_decode(
            self,
            sequences,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ):
            del clean_up_tokenization_spaces
            token_map = {0: "<pad>", 5: "A", 6: "B", 13: "<special>", 42: "C"}
            decoded = []
            for sequence in sequences:
                tokens = []
                for token in sequence:
                    token = int(token)
                    if skip_special_tokens and token == 13:
                        continue
                    if token == 0:
                        continue
                    tokens.append(token_map[token])
                decoded.append(" ".join(tokens))
            return decoded

        def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
            return self.batch_decode([ids], skip_special_tokens, clean_up_tokenization_spaces)[0]

    trainer = GOLDTrainer.__new__(GOLDTrainer)
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"), is_main_process=True)
    trainer.processing_class = RecordingTokenizer()
    trainer._tokenizer = RecordingTokenizer()
    trainer.args = SimpleNamespace(max_length=None, report_to=[])
    trainer.use_vllm = True
    trainer.use_uld_loss = False
    trainer.vllm_generation = RecordingVLLMGeneration()
    trainer.vllm_sync_frequency = 1
    trainer._last_vllm_sync_step = -1
    trainer.state = SimpleNamespace(global_step=0)
    trainer.num_generations = 1
    trainer.generation_config = SimpleNamespace(max_new_tokens=1)
    trainer._buffered_inputs = [None]
    trainer._buffered_text_logs = [None]

    slices = [
        {
            "slice": "original",
            "prompts": torch.tensor([[0, 0, 5, 13, 6]], dtype=torch.long),
            "prompt_attention_mask": torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.long),
        }
    ]

    GOLDTrainer._generate_on_policy_for_slices(trainer, slices, [0])

    buffered_inputs = trainer._buffered_inputs[0]
    assert trainer.vllm_generation.prompts == [[5, 13, 6]]
    assert trainer.vllm_generation.sync_calls == 1
    assert torch.equal(buffered_inputs["input_ids"], torch.tensor([[5, 13, 6, 42]], dtype=torch.long))
    assert torch.equal(
        buffered_inputs["attention_mask"],
        torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
    )
    assert torch.equal(
        buffered_inputs["labels"],
        torch.tensor([[-100, -100, -100, 42]], dtype=torch.long),
    )
    assert buffered_inputs["original_prompt_text"] == ["A <special> B"]
    assert buffered_inputs["original_completion_text"] == ["C"]
    assert trainer._buffered_text_logs[0] == (["A <special> B"], ["C"])


def test_on_policy_prompt_text_reflects_truncated_prompt():
    """When the prompt overflows max_length - max_completion_length it is truncated before the student sees it.
    `original_prompt_text` — which the teacher re-encodes — must reflect that truncated prompt, not the full one, so
    teacher and student score the completion under the same context."""

    class RecordingVLLMGeneration:
        def __init__(self):
            self.prompts = None

        def sync_weights(self):
            pass

        def generate(self, prompts, images, num_generations):
            self.prompts = prompts
            return None, [[42]], None, None

    class RecordingTokenizer:
        pad_token_id = 0
        pad_token = "<pad>"

        def __init__(self):
            self.truncation_side = "right"

        def batch_decode(
            self,
            sequences,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ):
            del clean_up_tokenization_spaces
            token_map = {0: "<pad>", 5: "A", 6: "B", 13: "<special>", 42: "C"}
            decoded = []
            for sequence in sequences:
                tokens = [token_map[int(t)] for t in sequence if int(t) != 0]
                decoded.append(" ".join(tokens))
            return decoded

        def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
            return self.batch_decode([ids], skip_special_tokens, clean_up_tokenization_spaces)[0]

    trainer = GOLDTrainer.__new__(GOLDTrainer)
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"), is_main_process=True)
    trainer.processing_class = RecordingTokenizer()
    trainer.args = SimpleNamespace(max_length=3, report_to=[])
    trainer.use_vllm = True
    trainer.use_uld_loss = False
    trainer.teacher_tokenizer = None
    trainer.uld_loss_fn = None
    trainer.vllm_generation = RecordingVLLMGeneration()
    trainer.vllm_sync_frequency = 1
    trainer._last_vllm_sync_step = -1
    trainer.state = SimpleNamespace(global_step=0)
    trainer.num_generations = 1
    trainer.generation_config = SimpleNamespace(max_new_tokens=1)
    trainer._buffered_inputs = [None]
    trainer._buffered_text_logs = [None]

    slices = [
        {
            "prompts": torch.tensor([[0, 0, 5, 13, 6]], dtype=torch.long),
            "prompt_attention_mask": torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.long),
        }
    ]

    GOLDTrainer._generate_on_policy_for_slices(trainer, slices, [0])

    buffered_inputs = trainer._buffered_inputs[0]
    # prompt_max_length = max_length - max_completion_length = 3 - 1 = 2; right-truncation keeps [5, 13].
    assert torch.equal(buffered_inputs["input_ids"], torch.tensor([[5, 13, 42]], dtype=torch.long))
    assert buffered_inputs["original_prompt_text"] == ["A <special>"]


def test_gold_trainer_init_defaults_vllm_max_model_length_to_max_length(monkeypatch):
    captured = {}

    class DummyStudentModel:
        def __init__(self):
            config = SimpleNamespace(_name_or_path="student", vocab_size=17)
            config.get_text_config = lambda: config
            self.config = config
            self.generation_config = SimpleNamespace(eos_token_id=2)
            self.name_or_path = "student"

    class DummyTeacherModel:
        def __init__(self):
            self.resized_to = None

        def resize_token_embeddings(self, vocab_size):
            self.resized_to = vocab_size

    class DummyProcessingClass:
        pad_token_id = 0

    def fake_sft_init(
        self,
        model,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=None,
        preprocess_logits_for_metrics=None,
        peft_config=None,
    ):
        del (
            data_collator,
            train_dataset,
            eval_dataset,
            compute_metrics,
            callbacks,
            optimizers,
        )
        del preprocess_logits_for_metrics, peft_config
        self.model = model
        self.args = args
        self.processing_class = processing_class
        self.accelerator = SimpleNamespace(
            device=torch.device("cpu"),
            num_processes=1,
            prepare_model=lambda module, evaluation_mode=True: module,
        )
        self.is_deepspeed_enabled = False
        self.is_fsdp_enabled = False

    class CapturingVLLMGeneration:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(gold_trainer_module.SFTTrainer, "__init__", fake_sft_init)
    monkeypatch.setattr(gold_trainer_module, "is_vllm_available", lambda: True)
    monkeypatch.setattr(gold_trainer_module, "VLLMGeneration", CapturingVLLMGeneration)

    args = SimpleNamespace(
        model_init_kwargs=None,
        max_length=128,
        use_liger_kernel=False,
        trust_remote_code=False,
        teacher_model_init_kwargs=None,
        use_uld_loss=False,
        teacher_tokenizer_name_or_path=None,
        teacher_model_revision=None,
        disable_dropout=False,
        lmbda=1.0,
        beta=0.5,
        temperature=1.0,
        top_p=1.0,
        seq_kd=False,
        num_generations=1,
        max_completion_length=16,
        top_k=0,
        log_completions=False,
        log_completions_steps=100,
        wandb_log_unique_prompts=True,
        num_completions_to_print=None,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_structured_outputs_regex=None,
        vllm_server_base_url=None,
        vllm_server_host="0.0.0.0",
        vllm_server_port=8001,
        vllm_group_port=51216,
        vllm_server_timeout=240.0,
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=0.2,
        vllm_max_model_length=None,
        vllm_enable_sleep_mode=False,
        vllm_model_impl="vllm",
        vllm_sync_frequency=1,
    )

    teacher_model = DummyTeacherModel()
    GOLDTrainer(
        model=DummyStudentModel(),
        teacher_model=teacher_model,
        args=args,
        data_collator=object(),
        processing_class=DummyProcessingClass(),
    )

    assert teacher_model.resized_to == 17
    assert captured["max_model_length"] == 128


def test_chatml_collator_truncates_keeping_completion_end(llama_tokenizer):
    """When the rendered chat-template message exceeds max_length, the collator must
    keep the LAST max_length tokens (the model's recent context), not the first. Also verifies byte_offsets are sliced
    consistently with input_ids."""
    long_user = "Please summarize:\n" + ("very long context. " * 200)  # well over 512 tokens
    long_assistant = "summary content goes here. " * 60
    examples = [
        {
            "messages": [
                {"role": "user", "content": long_user},
                {"role": "assistant", "content": long_assistant},
            ]
        }
    ]

    max_length = 512  # large enough to keep prompt tokens after the completion
    collator = DataCollatorForChatML(tokenizer=llama_tokenizer, max_length=max_length)
    batch = collator(examples)

    # Truncation must produce exactly max_length tokens (no left padding when full)
    assert batch["input_ids"].shape[1] == max_length

    # The last token of the kept sequence must match the EOS / final assistant token
    # of the full untruncated tokenization — proving we kept the END of the completion.
    backend = llama_tokenizer.backend_tokenizer
    formatted_message = llama_tokenizer.apply_chat_template(
        examples[0]["messages"], add_generation_prompt=False, tokenize=False
    )
    [(full_ids, _)] = encode_with_byte_offsets(backend, [formatted_message], add_special_tokens=False)
    assert batch["input_ids"][0, -1].item() == full_ids[-1]
    assert tuple(batch["byte_offsets"][0, -1].tolist())[1] > 0  # last completion-relative offset is non-zero


def test_chatml_collator_raises_when_completion_fills_window(llama_tokenizer):
    """When the completion alone fills the whole window, the prompt is entirely dropped, leaving nothing to
    generate from. The collator must reject this rather than emit an all-padding prompt row."""
    long_user = "Please summarize:\n" + ("very long context. " * 200)
    long_assistant = "summary content goes here. " * 60
    examples = [
        {
            "messages": [
                {"role": "user", "content": long_user},
                {"role": "assistant", "content": long_assistant},
            ]
        }
    ]

    max_length = 256  # smaller than the completion, so no prompt tokens survive
    collator = DataCollatorForChatML(tokenizer=llama_tokenizer, max_length=max_length)
    with pytest.raises(ValueError, match="no prompt tokens left after truncation"):
        collator(examples)


def test_prepared_tokenized_rows_keep_completion_after_truncation(llama_tokenizer):
    """When a GOLD row's prompt overflows max_length, dataset prep must keep the LAST max_length tokens (the
    completion end), tracking the prompt/completion boundary via completion_mask so the collator labels the completion
    instead of masking the whole sequence."""
    long_user = "Please summarize:\n" + ("very long context. " * 200)  # prompt alone overflows max_length
    assistant = "the short answer"
    dataset = Dataset.from_dict(
        {
            "messages": [
                [
                    {"role": "user", "content": long_user},
                    {"role": "assistant", "content": assistant},
                ]
            ]
        }
    )

    max_length = 64
    args = SimpleNamespace(
        dataset_num_proc=None,
        dataset_text_field="text",
        max_length=max_length,
        packing_strategy="bfd",
        use_liger_kernel=False,
        use_extended_uld=True,
    )
    trainer = GOLDTrainer.__new__(GOLDTrainer)
    prepared = trainer._prepare_dataset_with_original_text(
        dataset,
        llama_tokenizer,
        args,
        packing=False,
        formatting_func=None,
        dataset_name="train",
    )
    row = prepared[0]

    assert len(row["input_ids"]) == max_length  # truncated, not dropped
    assert 1 in row["completion_mask"]  # completion survived front-truncation

    # original_prompt_text / original_completion_text must reflect the truncated ids the student kept,
    # not the pre-truncation strings (otherwise the teacher would re-encode a longer prompt context).
    completion_start = row["completion_mask"].index(1)
    decode = partial(
        llama_tokenizer.decode,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    assert row["original_prompt_text"] == decode(row["input_ids"][:completion_start])
    assert row["original_completion_text"] == decode(row["input_ids"][completion_start:])

    collator = DataCollatorForChatML(tokenizer=llama_tokenizer, max_length=max_length)
    batch = collator([row])

    # The collator labels exactly the tracked completion tokens — never all -100 — and the assistant text survives.
    completion_ids = [tid for tid, m in zip(row["input_ids"], row["completion_mask"], strict=False) if m == 1]
    supervised = [label for label in batch["labels"][0].tolist() if label != -100]
    assert supervised == completion_ids
    assert assistant in llama_tokenizer.decode(completion_ids)


def test_prepared_tokenized_rows_rebase_byte_offsets_when_truncation_eats_into_completion(
    llama_tokenizer,
):
    """When truncation drops the front of the completion (``drop > completion_start``), the kept byte_offsets
    reference bytes in the original completion text — but ``original_completion_text`` is decoded fresh from the kept
    ids and starts at byte 0. The kept offsets must be rebased so they match the teacher's re-encoding.
    """
    short_prompt = "Q:"
    long_completion = "word " * 300  # completion alone overflows max_length
    dataset = Dataset.from_dict({"prompt": [short_prompt], "completion": [long_completion]})

    max_length = 32
    args = SimpleNamespace(
        dataset_num_proc=None,
        dataset_text_field="text",
        max_length=max_length,
        packing_strategy="bfd",
        use_liger_kernel=False,
        use_extended_uld=True,
    )
    trainer = GOLDTrainer.__new__(GOLDTrainer)
    prepared = trainer._prepare_dataset_with_original_text(
        dataset,
        llama_tokenizer,
        args,
        packing=False,
        formatting_func=None,
        dataset_name="train",
    )
    row = prepared[0]

    assert len(row["input_ids"]) == max_length
    # Prompt is so short that truncation ate into the completion: no prompt tokens survive.
    assert row["completion_mask"] == [1] * max_length

    # First kept completion token must start at byte 0 of the new (truncated) original_completion_text.
    assert tuple(row["byte_offsets"][0]) == (0, len(b"word "))


def test_prepare_dataset_messages_uses_last_assistant_turn(qwen_tokenizer):
    messages = [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "First?"},
        {"role": "assistant", "content": "One."},
        {"role": "user", "content": "Second?"},
        {"role": "assistant", "content": "Two."},
    ]
    dataset = Dataset.from_dict({"messages": [messages]})
    args = SimpleNamespace(
        dataset_num_proc=None,
        dataset_text_field="text",
        max_length=512,
        packing_strategy="bfd",
        use_liger_kernel=False,
        use_extended_uld=True,
    )
    trainer = GOLDTrainer.__new__(GOLDTrainer)

    prepared = trainer._prepare_dataset_with_original_text(
        dataset,
        qwen_tokenizer,
        args,
        packing=False,
        formatting_func=None,
        dataset_name="train",
    )
    row = prepared[0]
    expected_prompt = qwen_tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=False)
    expected_full = qwen_tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)

    assert row["original_prompt_text"] == expected_prompt
    assert row["original_completion_text"] == expected_full[len(expected_prompt) :]
    assert "One." not in row["original_completion_text"]
    assert "Two." in row["original_completion_text"]

    completion_ids = [tid for tid, mask in zip(row["input_ids"], row["completion_mask"], strict=True) if mask == 1]
    decoded_completion = qwen_tokenizer.decode(
        completion_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    assert decoded_completion == row["original_completion_text"]


def test_split_text_for_gold_preserves_source_text(gemma4_tokenizer):
    text = "Socrates is a man. All men are mortal."
    encoding = gemma4_tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)

    split = split_text_for_gold({"text": text}, gemma4_tokenizer, max_length=None)

    assert split["completion"].startswith(" ")
    assert split["prompt"] + split["completion"] == text
    assert (
        gemma4_tokenizer(split["prompt"] + split["completion"], add_special_tokens=False)["input_ids"]
        == encoding["input_ids"]
    )

    max_length = 4
    split = split_text_for_gold({"text": text}, gemma4_tokenizer, max_length=max_length)
    kept_offsets = encoding["offset_mapping"][-max_length:]
    assert split["prompt"] + split["completion"] == text[kept_offsets[0][0] : kept_offsets[-1][1]]


def test_prepare_dataset_extended_uld_keeps_seam_token(qwen_tokenizer):
    dataset = Dataset.from_dict({"prompt": ["Question: "], "completion": ["Answer."]})
    args = SimpleNamespace(
        dataset_num_proc=None,
        dataset_text_field="text",
        max_length=64,
        packing_strategy="bfd",
        use_liger_kernel=False,
        use_extended_uld=True,
    )
    trainer = GOLDTrainer.__new__(GOLDTrainer)

    row = trainer._prepare_dataset_with_original_text(
        dataset,
        qwen_tokenizer,
        args,
        packing=False,
        formatting_func=None,
        dataset_name="train",
    )[0]

    completion_ids = [
        token_id for token_id, mask in zip(row["input_ids"], row["completion_mask"], strict=True) if mask == 1
    ]
    completion_offsets = [
        offset for offset, mask in zip(row["byte_offsets"], row["completion_mask"], strict=True) if mask == 1
    ]
    assert row["original_completion_text"] == "Answer."
    assert qwen_tokenizer.decode(completion_ids[:-1]).lstrip() == row["original_completion_text"]
    assert completion_ids.count(qwen_tokenizer.eos_token_id) == 1
    assert completion_offsets[0] == [0, len(b"Answer")]
    assert completion_offsets[-1] == [len(b"Answer."), len(b"Answer.")]

    teacher_input_ids, teacher_labels, _, _ = build_teacher_inputs_from_texts(
        qwen_tokenizer,
        [row["original_prompt_text"]],
        [row["original_completion_text"]],
        use_extended_uld=True,
    )
    teacher_completion_ids = teacher_input_ids[0][teacher_labels[0] != -100].tolist()
    assert teacher_completion_ids.count(qwen_tokenizer.eos_token_id) == 1


def test_prepare_dataset_positional_uld_supports_sentencepiece(gemma4_tokenizer, qwen_tokenizer):
    dataset = Dataset.from_dict({"text": ["Question: Answer."], "prompt": ["Question: "], "completion": ["Answer."]})
    args = SimpleNamespace(
        dataset_num_proc=None,
        dataset_text_field="text",
        max_length=64,
        packing_strategy="bfd",
        use_liger_kernel=False,
        use_extended_uld=False,
    )
    trainer = GOLDTrainer.__new__(GOLDTrainer)

    prepared = trainer._prepare_dataset_with_original_text(
        dataset,
        gemma4_tokenizer,
        args,
        packing=False,
        formatting_func=None,
        dataset_name="train",
    )
    row = prepared[0]

    completion_ids = [
        token_id for token_id, mask in zip(row["input_ids"], row["completion_mask"], strict=True) if mask == 1
    ]
    assert row["original_completion_text"] == "Answer."
    assert completion_ids[-1] == gemma4_tokenizer.eos_token_id
    assert gemma4_tokenizer.decode(completion_ids[:-1]).lstrip() == row["original_completion_text"]
    assert row["byte_offsets"] == [[0, 0]] * len(row["input_ids"])

    teacher_input_ids, teacher_labels, _, _ = build_teacher_inputs_from_texts(
        qwen_tokenizer,
        [row["original_prompt_text"]],
        [row["original_completion_text"]],
        use_extended_uld=False,
    )
    teacher_completion_ids = teacher_input_ids[0][teacher_labels[0] != -100].tolist()
    assert qwen_tokenizer.decode(teacher_completion_ids) == "Answer." + qwen_tokenizer.eos_token


def test_build_teacher_inputs_positional_uld_supports_sentencepiece(gemma4_tokenizer):
    input_ids, labels, _, byte_offsets = build_teacher_inputs_from_texts(
        gemma4_tokenizer,
        ["Question: "],
        ["Answer."],
        use_extended_uld=False,
    )

    completion_ids = input_ids[0][labels[0] != -100].tolist()
    assert completion_ids.count(gemma4_tokenizer.eos_token_id) == 1
    assert gemma4_tokenizer.decode(completion_ids) == "Answer." + gemma4_tokenizer.eos_token
    assert byte_offsets.tolist() == [[[0, 0]] * input_ids.shape[1]]


def test_alignment_groups_cover_all_tokens(llama_tokenizer, qwen_tokenizer):
    config = build_config()
    loss = ULDLoss(config, student_tokenizer=llama_tokenizer, teacher_tokenizer=qwen_tokenizer)

    text = "SmolLM3-3B says hi 😊 to 你好."
    [(student_ids, student_offs)] = encode_with_byte_offsets(
        llama_tokenizer.backend_tokenizer, [text], add_special_tokens=False
    )
    [(teacher_ids, teacher_offs)] = encode_with_byte_offsets(
        qwen_tokenizer.backend_tokenizer, [text], add_special_tokens=False
    )

    student_groups, teacher_groups = loss._align_by_byte_offsets(student_offs, teacher_offs)

    assert len(student_groups) == len(teacher_groups)
    assert sorted(idx for group in student_groups for idx in group) == list(range(len(student_ids)))
    assert sorted(idx for group in teacher_groups for idx in group) == list(range(len(teacher_ids)))
    for student_group, teacher_group in zip(student_groups, teacher_groups, strict=True):
        student_span = (
            student_offs[student_group[0]][0],
            student_offs[student_group[-1]][1],
        )
        teacher_span = (
            teacher_offs[teacher_group[0]][0],
            teacher_offs[teacher_group[-1]][1],
        )
        assert student_span == teacher_span


def test_on_policy_completion_byte_offsets_match_encode_offsets(smollm_tokenizer, qwen_tokenizer):
    """On-policy offsets are derived from the generated token ids directly (per-token piece byte length, no
    decode-then-re-encode round-trip). For ByteLevel BPE — the family of every cross-tokenizer pair GOLD targets
    (SmolLM, Qwen, Llama 3+, …) — those per-token spans match what `encode_with_byte_offsets` produces on the same
    text, so student and teacher offsets share one byte coordinate system."""
    trainer = GOLDTrainer.__new__(GOLDTrainer)
    trainer.use_uld_loss = True
    trainer.teacher_tokenizer = qwen_tokenizer
    trainer.uld_loss_fn = SimpleNamespace(use_extended_uld=True)
    trainer.processing_class = smollm_tokenizer
    trainer._tokenizer = smollm_tokenizer
    trainer._is_vlm = False

    completion_text = "hello 你好 😊"
    [(completion_ids, expected_offsets)] = encode_with_byte_offsets(
        smollm_tokenizer.backend_tokenizer, [completion_text], add_special_tokens=False
    )
    input_ids = [smollm_tokenizer.pad_token_id] + completion_ids
    labels = [-100] + completion_ids
    updated_slice = {
        "input_ids": torch.tensor([input_ids]),
        "labels": torch.tensor([labels]),
    }

    trainer._maybe_add_completion_byte_offsets(updated_slice)

    assert [tuple(offset) for offset in updated_slice["byte_offsets"][0, 1:].tolist()] == expected_offsets


def test_merge_probabilities_multiplies_split_tokens_observed():
    config = build_config(uld_token_merge_strategy="observed")
    # Use simple 3-token vocabulary to validate merging behaviour
    # probs[0] = marginal P(token | context) at position 0 for all vocab tokens
    # probs[1] = P(token | context, token_0) at position 1 for all vocab tokens
    probs = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
    loss = ULDLoss(config, student_tokenizer=None, teacher_tokenizer=None)

    # token_ids[1] = 1 means the actual token at position 1 is token ID 1
    # So we should extract P(token_id=1 | ...) = probs[1, 1] = 0.5
    token_ids = [0, 1]  # Actual generated tokens

    merged = loss._merge_probabilities_with_alignment_groups(probs, [[0, 1]], token_ids=token_ids)

    # Expected: first position's marginal distribution × scalar conditional prob of actual token at later position
    # P_merged(y) = probs[0](y) × probs[1, token_ids[1]] = probs[0] × probs[1, 1]
    expected = probs[0] * probs[1, 1]  # probs[1, 1] = 0.5
    # Expected unnormalized: [0.6 * 0.5, 0.3 * 0.5, 0.1 * 0.5] = [0.30, 0.15, 0.05]

    torch.testing.assert_close(merged[0], expected)


def test_merge_probabilities_multiplies_split_tokens_bayesian():
    config = build_config(uld_token_merge_strategy="bayesian")
    # Use simple 3-token vocabulary to validate merging behaviour
    # probs[0] = P(token | context) at position 0 for all vocab tokens
    # probs[1] = P(token | context, token_0) at position 1 for all vocab tokens
    probs = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
    loss = ULDLoss(config, student_tokenizer=None, teacher_tokenizer=None)

    # token_ids[0] = 0 means the actual token at position 0 is token ID 0
    token_ids = [0, 1]  # Actual generated tokens

    merged = loss._merge_probabilities_with_alignment_groups(probs, [[0, 1]], token_ids=token_ids)

    # Expected: last position's full distribution × scalar prob of actual token at earlier position
    # P_merged(y) = probs[1](y) × probs[0, token_ids[0]] = probs[1] × probs[0, 0]
    expected = probs[1] * probs[0, 0]  # probs[0, 0] = 0.6
    # Expected unnormalized: [0.2 * 0.6, 0.5 * 0.6, 0.3 * 0.6] = [0.12, 0.30, 0.18]

    torch.testing.assert_close(merged[0], expected)


def test_compute_distillation_loss_bayesian_shifts_answer_logits():
    # The "bayesian" strategy shifts the answer-logit slice one position earlier (probs[k] predicts token_ids[k]).
    # This also applies on the positional path (use_extended_uld=False), so check the loss uses the shifted slice.
    config = build_config(use_extended_uld=False, uld_token_merge_strategy="bayesian")
    loss_fn = ULDLoss(config, student_tokenizer=None, teacher_tokenizer=None)

    torch.manual_seed(0)
    student_logits = torch.randn(1, 3, 4)
    teacher_logits = torch.randn(1, 3, 4)
    # Answer span starts at position 1 (size 2), leaving position 0 as the context the shift relies on.
    student_labels = torch.tensor([[-100, 1, 2]])
    teacher_labels = torch.tensor([[-100, 1, 2]])
    student_input_ids = torch.tensor([[0, 1, 2]])
    teacher_input_ids = torch.tensor([[0, 1, 2]])

    result = loss_fn._compute_distillation_loss(
        student_logits, teacher_logits, student_labels, teacher_labels, student_input_ids, teacher_input_ids
    )

    # Expected: probabilities come from the shifted positions [0, 1], not the answer positions [1, 2].
    student_probs = torch.softmax(student_logits[0, 0:2], dim=-1)
    teacher_probs = torch.softmax(teacher_logits[0, 0:2], dim=-1)
    student_sorted = student_probs.sort(dim=-1, descending=True).values
    teacher_sorted = teacher_probs.sort(dim=-1, descending=True).values
    expected = (student_sorted - teacher_sorted).abs().sum() / student_probs.size(0)

    torch.testing.assert_close(result, expected)

    # The "observed" strategy uses the unshifted answer positions [1, 2], so it gives a different loss.
    observed_fn = ULDLoss(build_config(use_extended_uld=False), student_tokenizer=None, teacher_tokenizer=None)
    observed = observed_fn._compute_distillation_loss(
        student_logits, teacher_logits, student_labels, teacher_labels, student_input_ids, teacher_input_ids
    )
    assert not torch.allclose(observed, expected)


def test_compute_distillation_loss_bayesian_skips_zero_start_span_positional():
    # Front-truncation can drop the whole prompt, leaving an answer span that starts at index 0. The first token has
    # no preceding predictor logit for the shift, so it must be skipped (start += 1, size -= 1) rather than wrapping
    # around with negative indexing.
    config = build_config(use_extended_uld=False, uld_token_merge_strategy="bayesian")
    loss_fn = ULDLoss(config, student_tokenizer=None, teacher_tokenizer=None)

    torch.manual_seed(0)
    student_logits = torch.randn(1, 3, 4)
    teacher_logits = torch.randn(1, 3, 4)
    # No -100 prefix: the answer span starts at index 0 with size 3.
    student_labels = torch.tensor([[1, 2, 3]])
    teacher_labels = torch.tensor([[1, 2, 3]])
    student_input_ids = torch.tensor([[1, 2, 3]])
    teacher_input_ids = torch.tensor([[1, 2, 3]])

    result = loss_fn._compute_distillation_loss(
        student_logits, teacher_logits, student_labels, teacher_labels, student_input_ids, teacher_input_ids
    )

    # After skipping the leading token: start=1, size=2 -> the shifted slice covers positions [0, 1].
    student_probs = torch.softmax(student_logits[0, 0:2], dim=-1)
    teacher_probs = torch.softmax(teacher_logits[0, 0:2], dim=-1)
    student_sorted = student_probs.sort(dim=-1, descending=True).values
    teacher_sorted = teacher_probs.sort(dim=-1, descending=True).values
    expected = (student_sorted - teacher_sorted).abs().sum() / student_probs.size(0)

    torch.testing.assert_close(result, expected)


def test_compute_distillation_loss_bayesian_zero_loss_when_only_token_dropped():
    # If the answer span is a single token at index 0, skipping it leaves nothing to score -> zero loss, no crash.
    config = build_config(use_extended_uld=False, uld_token_merge_strategy="bayesian")
    loss_fn = ULDLoss(config, student_tokenizer=None, teacher_tokenizer=None)

    student_logits = torch.randn(1, 2, 4)
    teacher_logits = torch.randn(1, 2, 4)
    student_labels = torch.tensor([[5, -100]])  # start 0, size 1
    teacher_labels = torch.tensor([[5, -100]])
    student_input_ids = torch.tensor([[5, 0]])
    teacher_input_ids = torch.tensor([[5, 0]])

    result = loss_fn._compute_distillation_loss(
        student_logits, teacher_logits, student_labels, teacher_labels, student_input_ids, teacher_input_ids
    )

    torch.testing.assert_close(result, torch.zeros_like(result))


def test_compute_distillation_loss_bayesian_skips_zero_start_span_extended():
    # Same zero-start scenario on the extended (byte-offset) path: the first aligned group pair is dropped instead of
    # wrapping around, so the loss is finite.
    config = build_config(use_extended_uld=True, uld_token_merge_strategy="bayesian")
    loss_fn = ULDLoss(config, student_tokenizer=None, teacher_tokenizer=None)

    torch.manual_seed(0)
    student_logits = torch.randn(1, 3, 4)
    teacher_logits = torch.randn(1, 3, 4)
    student_labels = torch.tensor([[1, 2, 3]])  # start 0, size 3
    teacher_labels = torch.tensor([[1, 2, 3]])
    student_input_ids = torch.tensor([[1, 2, 3]])
    teacher_input_ids = torch.tensor([[1, 2, 3]])
    # 1:1 byte alignment over the completion: each token spans one byte.
    byte_offsets = torch.tensor([[[0, 1], [1, 2], [2, 3]]])

    result = loss_fn._compute_distillation_loss(
        student_logits,
        teacher_logits,
        student_labels,
        teacher_labels,
        student_input_ids,
        teacher_input_ids,
        student_byte_offsets=byte_offsets,
        teacher_byte_offsets=byte_offsets,
    )

    assert torch.isfinite(result)


def test_uldloss_positional_mode_does_not_require_byte_offsets():
    config = build_config(use_extended_uld=False)
    loss_fn = ULDLoss(config, student_tokenizer=None, teacher_tokenizer=None)

    student_logits = torch.randn(1, 4, 5)
    teacher_logits = torch.randn(1, 4, 6)
    student_labels = torch.tensor([[-100, 1, 2, -100]])
    teacher_labels = torch.tensor([[-100, 3, 4, -100]])
    student_input_ids = torch.tensor([[0, 1, 2, 0]])
    teacher_input_ids = torch.tensor([[0, 3, 4, 0]])

    loss = loss_fn(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        student_labels=student_labels,
        teacher_labels=teacher_labels,
        student_input_ids=student_input_ids,
        teacher_input_ids=teacher_input_ids,
    )

    assert torch.isfinite(loss)


def test_initialize_vocabulary_mapping_contains_common_tokens(llama_tokenizer, qwen_tokenizer):
    config = build_config(
        uld_use_hybrid_loss=True,
        uld_hybrid_matched_weight=1.0,
        uld_hybrid_unmatched_weight=0.0,
    )
    loss = ULDLoss(config, student_tokenizer=llama_tokenizer, teacher_tokenizer=qwen_tokenizer)

    common_tokens = ["Hello", "world", "-", "ol", "LM", "3", "B"]
    for token in common_tokens:
        student_id = llama_tokenizer.convert_tokens_to_ids(token)
        teacher_id = qwen_tokenizer.convert_tokens_to_ids(token)
        assert student_id is not None
        assert teacher_id is not None
        assert teacher_id in loss._vocab_mapping
        assert loss._vocab_mapping[teacher_id] == student_id
        assert teacher_id in loss._teacher_matched_ids
        assert student_id in loss._student_matched_ids


def test_get_start_and_size_answers_skips_prompt_tokens():
    trainer = ULDLoss.__new__(ULDLoss)
    trainer.ignore_index = -100

    answers = torch.tensor(
        [
            [-100, -100, -100, 10, 20, 30, -100, -100],
            [-100, 5, 6, 7, -100, -100, -100, -100],
            [-100, -100, -100, -100, -100, -100, -100, -100],
        ]
    )

    starts, sizes = trainer._get_start_and_size_answers(answers)

    assert starts == [3, 1, 0]
    assert sizes == [3, 3, 0]


@pytest.mark.slow
def test_generate_on_policy_outputs_masks_prompt(llama_tokenizer):
    trainer = GOLDTrainer.__new__(GOLDTrainer)
    trainer.processing_class = llama_tokenizer

    prompt_text = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nHello?<|eot_id|>"
    completion_text = "<|start_header_id|>assistant<|end_header_id|>\nHi there!"

    prompt_ids = llama_tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    completion_ids = llama_tokenizer(completion_text, add_special_tokens=False)["input_ids"]

    pad_id = llama_tokenizer.pad_token_id
    pad_width = 3
    prompt_tensor = torch.full((1, len(prompt_ids) + pad_width), pad_id, dtype=torch.long)
    prompt_tensor[0, pad_width:] = torch.tensor(prompt_ids, dtype=torch.long)
    prompt_mask = (prompt_tensor != pad_id).long()

    # model.generate() returns full sequences including left-padding from the input
    completion_tensor = torch.tensor(completion_ids, dtype=torch.long).unsqueeze(0)
    generated_sequence = torch.cat([prompt_tensor, completion_tensor], dim=1)

    class DummyModel:
        def generate(self, input_ids, attention_mask, generation_config, return_dict_in_generate):
            assert torch.equal(input_ids, prompt_tensor)
            assert torch.equal(attention_mask, prompt_mask)
            return SimpleNamespace(sequences=generated_sequence)

    generation_config = SimpleNamespace(max_completion_length=None, temperature=None, top_k=None, top_p=None)
    new_ids, new_mask, new_labels, prompt_texts, completion_texts = GOLDTrainer.generate_on_policy_outputs(
        trainer,
        DummyModel(),
        {"prompts": prompt_tensor, "prompt_attention_mask": prompt_mask},
        generation_config,
        pad_id,
    )

    assert torch.equal(new_ids, generated_sequence)
    if pad_id is not None:
        expected_mask = (generated_sequence != pad_id).long()
        assert torch.equal(new_mask, expected_mask)
    else:
        assert torch.all(new_mask == 1)

    padded_prompt_len = prompt_tensor.shape[1]
    assert torch.all(new_labels[0, :padded_prompt_len] == -100)
    assert torch.equal(
        new_labels[0, padded_prompt_len:],
        torch.tensor(completion_ids, dtype=torch.long),
    )

    unpadded_prompt_ids = prompt_tensor[0][prompt_mask[0].bool()].tolist()
    assert prompt_texts[0] == llama_tokenizer.decode(
        unpadded_prompt_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    assert completion_texts[0] == llama_tokenizer.decode(
        completion_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )


@pytest.mark.slow
def test_generate_on_policy_outputs_masks_prompt_smollm(smollm_tokenizer, openr1_examples):
    trainer = GOLDTrainer.__new__(GOLDTrainer)
    trainer.processing_class = smollm_tokenizer

    collator = DataCollatorForChatML(tokenizer=smollm_tokenizer)
    batch = collator([openr1_examples[0]])
    batch = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    class DummyModel:
        def generate(self, input_ids, attention_mask, generation_config, return_dict_in_generate):
            assert torch.equal(input_ids, batch["prompts"])
            assert torch.equal(attention_mask, batch["prompt_attention_mask"])
            return SimpleNamespace(sequences=batch["input_ids"])

    generation_config = SimpleNamespace(max_completion_length=None, temperature=None, top_k=None, top_p=None)
    pad_id = smollm_tokenizer.pad_token_id
    new_ids, new_mask, new_labels, prompt_texts, completion_texts = GOLDTrainer.generate_on_policy_outputs(
        trainer,
        DummyModel(),
        {
            "prompts": batch["prompts"],
            "prompt_attention_mask": batch["prompt_attention_mask"],
        },
        generation_config,
        pad_id,
    )

    assert torch.equal(new_ids, batch["input_ids"])
    if pad_id is not None:
        expected_mask = (batch["input_ids"] != pad_id).long()
        assert torch.equal(new_mask, expected_mask)
    else:
        assert torch.all(new_mask == 1)

    prompt_len = int(batch["prompt_attention_mask"].sum().item())
    tail_labels = new_labels[0, prompt_len:]
    expected_tail = batch["input_ids"][0, prompt_len:]
    active_mask = tail_labels != -100
    assert torch.all(new_labels[0, :prompt_len] == -100)
    assert torch.equal(tail_labels[active_mask], expected_tail[active_mask])
    assert torch.all(tail_labels[~active_mask] == -100)

    prompt_tokens = batch["prompts"][0, batch["prompt_attention_mask"][0].bool()]
    decoded_prompt = smollm_tokenizer.decode(prompt_tokens.tolist(), skip_special_tokens=False)
    assert prompt_texts[0] == decoded_prompt

    assistant_completion = openr1_examples[0]["messages"][-1]["content"].strip()
    assert assistant_completion in completion_texts[0]


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

    torch.testing.assert_close(loss, expected)


def test_uldloss_handles_llama_student_qwen_teacher_sequence(llama_tokenizer, qwen_tokenizer):
    config = build_config(
        uld_use_hybrid_loss=True,
        uld_hybrid_matched_weight=0.6,
        uld_hybrid_unmatched_weight=0.4,
    )
    loss_fn = ULDLoss(config, student_tokenizer=llama_tokenizer, teacher_tokenizer=qwen_tokenizer)

    prompt = "User: Summarize the difference between llamas and alpacas."
    completion = "Assistant: Llamas are taller while alpacas have softer wool."

    student_ids, student_labels, student_offsets = encode_prompt_completion(llama_tokenizer, prompt, completion)
    teacher_ids, teacher_labels, teacher_offsets = encode_prompt_completion(qwen_tokenizer, prompt, completion)

    pad_id_student = llama_tokenizer.pad_token_id
    pad_id_teacher = qwen_tokenizer.pad_token_id
    max_length = max(len(student_ids), len(teacher_ids))

    student_ids = pad_tokens(student_ids, pad_id_student, max_length)
    teacher_ids = pad_tokens(teacher_ids, pad_id_teacher, max_length)
    student_labels = pad_labels(student_labels, max_length)
    teacher_labels = pad_labels(teacher_labels, max_length)
    student_byte_offsets = pad_byte_offsets(student_offsets, max_length, padding_side="right").unsqueeze(0)
    teacher_byte_offsets = pad_byte_offsets(teacher_offsets, max_length, padding_side="right").unsqueeze(0)

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
        student_byte_offsets=student_byte_offsets,
        teacher_byte_offsets=teacher_byte_offsets,
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

    student_ids, student_labels, student_offsets = encode_prompt_completion(smollm_tokenizer, prompt, completion)
    teacher_ids, teacher_labels, teacher_offsets = encode_prompt_completion(qwen_tokenizer, prompt, completion)

    pad_id_student = smollm_tokenizer.pad_token_id
    pad_id_teacher = qwen_tokenizer.pad_token_id
    max_length = max(len(student_ids), len(teacher_ids))

    student_ids = pad_tokens(student_ids, pad_id_student, max_length)
    teacher_ids = pad_tokens(teacher_ids, pad_id_teacher, max_length)
    student_labels = pad_labels(student_labels, max_length)
    teacher_labels = pad_labels(teacher_labels, max_length)
    student_byte_offsets = pad_byte_offsets(student_offsets, max_length, padding_side="right").unsqueeze(0)
    teacher_byte_offsets = pad_byte_offsets(teacher_offsets, max_length, padding_side="right").unsqueeze(0)

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
        student_byte_offsets=student_byte_offsets,
        teacher_byte_offsets=teacher_byte_offsets,
    )

    assert torch.isfinite(loss)
    assert loss.dim() == 0
    assert loss_fn.last_matched_loss is not None
    assert loss_fn.last_unmatched_loss is not None


def test_uldloss_hybrid_config_beta_zero(llama_tokenizer, qwen_tokenizer):
    config = build_config(
        uld_use_hybrid_loss=True,
        uld_hybrid_matched_weight=0.0,
        uld_hybrid_unmatched_weight=1.0,
        uld_crossentropy_weight=0.0,
        uld_distillation_weight=1.0,
        uld_student_temperature=1.0,
        uld_teacher_temperature=1.0,
        temperature=1.0,
        top_p=0.95,
        top_k=0,
        lmbda=1.0,
        beta=0.0,
    )
    loss_fn = ULDLoss(config, student_tokenizer=llama_tokenizer, teacher_tokenizer=qwen_tokenizer)

    prompt = "User: Explain how GOLD handles tokenizer mismatches."
    completion = "Assistant: GOLD merges aligned subwords and applies hybrid ULD loss."

    student_ids, student_labels, student_offsets = encode_prompt_completion(llama_tokenizer, prompt, completion)
    teacher_ids, teacher_labels, teacher_offsets = encode_prompt_completion(qwen_tokenizer, prompt, completion)

    pad_id_student = llama_tokenizer.pad_token_id
    pad_id_teacher = qwen_tokenizer.pad_token_id
    max_length = max(len(student_ids), len(teacher_ids))

    student_ids = pad_tokens(student_ids, pad_id_student, max_length)
    teacher_ids = pad_tokens(teacher_ids, pad_id_teacher, max_length)
    student_labels = pad_labels(student_labels, max_length)
    teacher_labels = pad_labels(teacher_labels, max_length)
    student_byte_offsets = pad_byte_offsets(student_offsets, max_length, padding_side="right").unsqueeze(0)
    teacher_byte_offsets = pad_byte_offsets(teacher_offsets, max_length, padding_side="right").unsqueeze(0)

    student_input_ids = torch.tensor([student_ids])
    teacher_input_ids = torch.tensor([teacher_ids])
    student_labels = torch.tensor([student_labels])
    teacher_labels = torch.tensor([teacher_labels])

    student_vocab = len(llama_tokenizer)
    teacher_vocab = len(qwen_tokenizer)
    torch.manual_seed(0)
    student_logits = torch.randn(1, max_length, student_vocab)
    teacher_logits = torch.randn(1, max_length, teacher_vocab)

    loss = loss_fn(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        student_labels=student_labels,
        teacher_labels=teacher_labels,
        student_input_ids=student_input_ids,
        teacher_input_ids=teacher_input_ids,
        student_byte_offsets=student_byte_offsets,
        teacher_byte_offsets=teacher_byte_offsets,
    )

    assert torch.isfinite(loss)
    assert loss.dim() == 0
    assert loss_fn.last_matched_loss is not None
    assert loss_fn.last_unmatched_loss is not None

    expected = config.uld_hybrid_unmatched_weight * loss_fn.last_unmatched_loss
    torch.testing.assert_close(loss, expected, atol=1e-6, rtol=1e-5)


def test_vlm_alignment_groups_cover_all_tokens_smolvlm_qwen3vl(smolvlm_processor, qwen3_vl_processor, vlm_examples):
    student_tokenizer = smolvlm_processor.tokenizer
    teacher_tokenizer = qwen3_vl_processor.tokenizer

    collator = DataCollatorForVisionLanguageChatML(processor=smolvlm_processor, max_length=2048)
    batch = collator(vlm_examples)

    config = build_config()
    loss = ULDLoss(config, student_tokenizer=student_tokenizer, teacher_tokenizer=teacher_tokenizer)

    teacher_input_ids, teacher_labels, _, teacher_byte_offsets = _teacher_inputs_from_collator(
        student_tokenizer, teacher_tokenizer, batch
    )

    _assert_alignment_covers_completion(loss, batch, teacher_input_ids, teacher_labels, teacher_byte_offsets)


def test_build_teacher_vlm_inputs_feeds_images_and_completion_byte_offsets(qwen3_vl_processor, vlm_examples):
    """Cross-architecture VLM ULD must render the teacher prompt through the teacher processor (so it
    actually sees the image via pixel_values) while keeping completion byte offsets relative to the original completion
    text — the coordinate system shared with the student."""
    trainer = GOLDTrainer.__new__(GOLDTrainer)
    trainer._teacher_processor = qwen3_vl_processor
    trainer.teacher_tokenizer = qwen3_vl_processor.tokenizer
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))

    images, prompts = trainer._extract_images_and_prompts(vlm_examples)
    completion_texts = _get_assistant_texts(vlm_examples)

    (
        teacher_input_ids,
        teacher_labels,
        teacher_attention_mask,
        teacher_byte_offsets,
        teacher_forward_kwargs,
    ) = trainer._build_teacher_vlm_inputs(completion_texts, images, prompts)

    batch_size = len(vlm_examples)
    seq_len = teacher_input_ids.shape[1]
    assert teacher_input_ids.shape == teacher_labels.shape == teacher_attention_mask.shape
    assert teacher_byte_offsets.shape == (batch_size, seq_len, 2)
    # The teacher actually receives the image, not just text.
    assert "pixel_values" in teacher_forward_kwargs

    backend = qwen3_vl_processor.tokenizer.backend_tokenizer
    for row in range(batch_size):
        completion_positions = teacher_labels[row] != -100
        # The prompt (image placeholders + text) is masked; the completion is supervised.
        assert completion_positions.any()
        assert not completion_positions[0]

        expected_ids, expected_offs = encode_with_byte_offsets(backend, [completion_texts[row]])[0]
        # Completion ids carry their byte offsets; the appended EOS sits at the end of the content.
        content_len = len(completion_texts[row].encode("utf-8"))
        row_completion_ids = teacher_input_ids[row][completion_positions].tolist()
        row_completion_offs = teacher_byte_offsets[row][completion_positions].tolist()
        assert row_completion_ids[: len(expected_ids)] == expected_ids
        assert row_completion_offs[: len(expected_offs)] == [list(off) for off in expected_offs]
        assert row_completion_offs[-1] == [content_len, content_len]


def test_gold_trainer_init_rejects_llm_with_vision_dataset(monkeypatch):
    """GOLDTrainer should raise ValueError when a text-only model receives a vision dataset."""

    class DummyStudentModel:
        def __init__(self):
            self.config = SimpleNamespace(_name_or_path="student", vocab_size=17)
            self.generation_config = SimpleNamespace(eos_token_id=2)
            self.name_or_path = "student"

    class DummyTeacherModel:
        def __init__(self):
            self.resized_to = None

        def resize_token_embeddings(self, vocab_size):
            self.resized_to = vocab_size

    def fake_sft_init(
        self,
        model,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=None,
        preprocess_logits_for_metrics=None,
        peft_config=None,
    ):
        del (
            data_collator,
            train_dataset,
            eval_dataset,
            compute_metrics,
            callbacks,
            optimizers,
        )
        del preprocess_logits_for_metrics, peft_config
        self.model = model
        self.args = args
        self.processing_class = processing_class
        self.accelerator = SimpleNamespace(
            device=torch.device("cpu"),
            num_processes=1,
            prepare_model=lambda module, evaluation_mode=True: module,
        )
        self.is_deepspeed_enabled = False
        self.is_fsdp_enabled = False

    monkeypatch.setattr(gold_trainer_module.SFTTrainer, "__init__", fake_sft_init)

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset with an "image" key triggers vision detection
    vision_dataset = Dataset.from_dict({"messages": [["dummy"]], "image": ["fake_image"]})

    args = SimpleNamespace(
        model_init_kwargs=None,
        max_length=128,
        use_liger_kernel=False,
        trust_remote_code=False,
        teacher_model_init_kwargs=None,
        use_uld_loss=False,
        teacher_tokenizer_name_or_path=None,
        teacher_model_revision=None,
        disable_dropout=False,
        lmbda=1.0,
        beta=0.5,
        temperature=1.0,
        top_p=1.0,
        seq_kd=False,
        num_generations=1,
        max_completion_length=16,
        top_k=0,
        log_completions=False,
        log_completions_steps=100,
        wandb_log_unique_prompts=True,
        num_completions_to_print=None,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        use_vllm=False,
    )

    with pytest.raises(ValueError, match="vision-related"):
        GOLDTrainer(
            model=DummyStudentModel(),
            teacher_model=DummyTeacherModel(),
            args=args,
            train_dataset=vision_dataset,
            processing_class=tokenizer,
        )


def _get_assistant_texts(examples):
    """Extract assistant text content from examples, handling both plain string and multimodal format."""
    texts = []
    for example in examples:
        content = example["completion"][-1]["content"]
        if isinstance(content, list):
            texts.append("".join(part["text"] for part in content if "text" in part))
        else:
            texts.append(content)
    return texts


def _get_prompt_turn_texts(example):
    texts = []
    for turn in example["prompt"]:
        content = turn["content"]
        if isinstance(content, list):
            text = "\n".join(part["text"] for part in content if isinstance(part, dict) and "text" in part)
        else:
            text = content
        if text:
            texts.append(text)
    return texts


def test_vlm_chatml_collator_preserves_completion_smolvlm(smolvlm_processor, qwen3_vl_processor, vlm_examples):
    # 2048 to not truncate the completion tokens
    collator = DataCollatorForVisionLanguageChatML(processor=smolvlm_processor, max_length=2048)
    batch = collator(vlm_examples)

    # Verify basic batch structure
    assert "input_ids" in batch
    assert "labels" in batch
    assert "prompts" in batch
    assert "prompt_attention_mask" in batch
    assert "pixel_values" in batch
    assert "original_prompt_text" in batch
    assert "original_completion_text" in batch

    # Verify completions are preserved in decoded output
    assistant_texts = _get_assistant_texts(vlm_examples)
    decoded_batch = smolvlm_processor.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=False)
    for decoded, assistant in zip(decoded_batch, assistant_texts, strict=True):
        assert assistant in decoded

    # Verify ULD cross-tokenizer distillation with teacher inputs
    student_tokenizer = smolvlm_processor.tokenizer
    teacher_tokenizer = qwen3_vl_processor.tokenizer

    teacher_input_ids, teacher_labels, completion_texts, teacher_byte_offsets = _teacher_inputs_from_collator(
        student_tokenizer, teacher_tokenizer, batch
    )
    for completion, assistant in zip(completion_texts, assistant_texts, strict=True):
        assert assistant.strip() in completion
        assert completion.strip()

    config = build_config(
        uld_use_hybrid_loss=True,
        uld_hybrid_matched_weight=0.6,
        uld_hybrid_unmatched_weight=0.4,
    )
    loss_fn = ULDLoss(config, student_tokenizer=student_tokenizer, teacher_tokenizer=teacher_tokenizer)

    _assert_alignment_covers_completion(loss_fn, batch, teacher_input_ids, teacher_labels, teacher_byte_offsets)

    torch.manual_seed(42)
    student_vocab = len(student_tokenizer)
    teacher_vocab = len(teacher_tokenizer)
    batch_size, seq_len = batch["input_ids"].shape
    student_logits = torch.randn(batch_size, seq_len, student_vocab)
    teacher_logits = torch.randn(batch_size, teacher_input_ids.shape[1], teacher_vocab)

    loss = loss_fn(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        student_labels=batch["labels"],
        teacher_labels=teacher_labels,
        student_input_ids=batch["input_ids"],
        teacher_input_ids=teacher_input_ids,
        student_byte_offsets=batch["byte_offsets"],
        teacher_byte_offsets=teacher_byte_offsets,
    )

    assert torch.isfinite(loss)


@pytest.mark.slow
def test_vlm_chatml_collator_preserves_completion_qwen3vl(smolvlm_processor, qwen3_vl_processor, vlm_examples):
    collator = DataCollatorForVisionLanguageChatML(processor=qwen3_vl_processor, max_length=2048)
    batch = collator(vlm_examples)

    # Verify basic batch structure
    assert "input_ids" in batch
    assert "labels" in batch
    assert "prompts" in batch
    assert "pixel_values" in batch

    # Verify completions are preserved in decoded output
    assistant_texts = _get_assistant_texts(vlm_examples)
    decoded_batch = qwen3_vl_processor.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=False)
    for decoded, assistant in zip(decoded_batch, assistant_texts, strict=True):
        assert assistant in decoded

    # Verify ULD cross-tokenizer distillation with teacher inputs
    student_tokenizer = qwen3_vl_processor.tokenizer
    teacher_tokenizer = smolvlm_processor.tokenizer

    teacher_input_ids, teacher_labels, completion_texts, teacher_byte_offsets = _teacher_inputs_from_collator(
        student_tokenizer, teacher_tokenizer, batch
    )
    for completion, assistant in zip(completion_texts, assistant_texts, strict=True):
        assert assistant.strip() in completion
        assert completion.strip()

    config = build_config(
        uld_use_hybrid_loss=True,
        uld_hybrid_matched_weight=0.6,
        uld_hybrid_unmatched_weight=0.4,
    )
    loss_fn = ULDLoss(config, student_tokenizer=student_tokenizer, teacher_tokenizer=teacher_tokenizer)

    _assert_alignment_covers_completion(loss_fn, batch, teacher_input_ids, teacher_labels, teacher_byte_offsets)

    torch.manual_seed(43)
    student_vocab = len(student_tokenizer)
    teacher_vocab = len(teacher_tokenizer)
    batch_size, seq_len = batch["input_ids"].shape
    student_logits = torch.randn(batch_size, seq_len, student_vocab)
    teacher_logits = torch.randn(batch_size, teacher_input_ids.shape[1], teacher_vocab)

    loss = loss_fn(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        student_labels=batch["labels"],
        teacher_labels=teacher_labels,
        student_input_ids=batch["input_ids"],
        teacher_input_ids=teacher_input_ids,
        student_byte_offsets=batch["byte_offsets"],
        teacher_byte_offsets=teacher_byte_offsets,
    )

    assert torch.isfinite(loss)


def test_vlm_collator_label_masking_and_prompt_truncation(smolvlm_processor, vlm_examples):
    """Verify that the VLM collator:
    1. Masks prompt and padding tokens in labels, leaves completion tokens unmasked.
    2. Truncates `prompts`/`prompt_attention_mask` to `max_length` (keeping the start, matching SFT/DPO VLM truncation)
       so on-policy `model.generate(input_ids=inputs["prompts"])` never exceeds `max_length`."""
    collator = DataCollatorForVisionLanguageChatML(processor=smolvlm_processor, max_length=2048)
    batch = collator(copy.deepcopy(vlm_examples))

    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]

    for i in range(input_ids.shape[0]):
        # Padding tokens (attention_mask == 0) must be masked in labels
        padding_positions = attention_mask[i] == 0
        assert (labels[i][padding_positions] == -100).all(), "Padding tokens should be masked with -100"

        # There must be at least one non-masked label (completion token)
        completion_positions = labels[i] != -100
        assert completion_positions.any(), "Each example must have at least one completion token in labels"

        # Completion labels must match the corresponding input_ids
        assert (labels[i][completion_positions] == input_ids[i][completion_positions]).all(), (
            "Unmasked labels must match input_ids"
        )

        # Prompt tokens (attended but masked in labels) must exist — the prompt is never empty
        prompt_positions = (attention_mask[i] == 1) & (labels[i] == -100)
        assert prompt_positions.any(), "Each example must have masked prompt tokens"

    # Truncation: prompt tensors must be truncated to max_length, keeping the start
    full_prompt_len = batch["prompts"].shape[1]
    short_max_length = max(1, full_prompt_len - 1)
    collator_short = DataCollatorForVisionLanguageChatML(processor=smolvlm_processor, max_length=short_max_length)
    short_batch = collator_short(copy.deepcopy(vlm_examples))

    assert short_batch["prompts"].shape[1] <= short_max_length, (
        f"prompts tensor should be truncated to max_length={short_max_length}, "
        f"got shape[1]={short_batch['prompts'].shape[1]}"
    )
    assert short_batch["prompt_attention_mask"].shape[1] <= short_max_length, (
        f"prompt_attention_mask should be truncated to max_length={short_max_length}, "
        f"got shape[1]={short_batch['prompt_attention_mask'].shape[1]}"
    )
    # Keep the start of the tensor (preserves image tokens at the start of the prompt)
    assert torch.equal(short_batch["prompts"], batch["prompts"][:, :short_max_length])
    assert torch.equal(short_batch["prompt_attention_mask"], batch["prompt_attention_mask"][:, :short_max_length])


def test_vlm_collator_original_text_is_untemplated(smolvlm_processor, vlm_examples):
    """`original_*_text` must be free of the student's chat-template markers.

    Cross-tokenizer ULD distillation re-renders the prompt through the teacher's chat template and concatenates the
    stored completion. If the stored completion still carries the student's special tokens (e.g. ``<|im_end|>``, role
    headers), the teacher tokenizer will tokenize them as regular text, producing spurious teacher tokens and incorrect
    teacher logits.
    """
    collator = DataCollatorForVisionLanguageChatML(processor=smolvlm_processor, max_length=2048)
    batch = collator(copy.deepcopy(vlm_examples))

    student_tokenizer = smolvlm_processor.tokenizer
    student_specials = [tok for tok in student_tokenizer.all_special_tokens if tok and tok.strip()]

    assert "original_prompt_text" in batch
    assert "original_completion_text" in batch
    assert len(batch["original_prompt_text"]) == len(vlm_examples)
    assert len(batch["original_completion_text"]) == len(vlm_examples)

    expected_assistant_texts = _get_assistant_texts(vlm_examples)
    for raw_completion, assistant_text in zip(
        batch["original_completion_text"], expected_assistant_texts, strict=True
    ):
        assert assistant_text.strip() in raw_completion
        for special in student_specials:
            assert special not in raw_completion, (
                f"original_completion_text leaked student special token {special!r}: {raw_completion!r}"
            )

    for raw_prompt, example in zip(batch["original_prompt_text"], vlm_examples, strict=True):
        prompt_turn_texts = _get_prompt_turn_texts(example)
        for text in prompt_turn_texts:
            assert text.strip() in raw_prompt
        if len(prompt_turn_texts) > 1:
            assert "\n".join(prompt_turn_texts) == raw_prompt
            assert "".join(prompt_turn_texts) != raw_prompt
        for special in student_specials:
            assert special not in raw_prompt, (
                f"original_prompt_text leaked student special token {special!r}: {raw_prompt!r}"
            )


def test_gold_trainer_init_rejects_non_vlm_teacher(monkeypatch):
    """GOLDTrainer should raise ValueError when the student is a VLM but the teacher is not."""

    class DummyStudentModel:
        def __init__(self):
            self.config = SimpleNamespace(_name_or_path="student", vocab_size=17)
            self.generation_config = SimpleNamespace(eos_token_id=2)
            self.name_or_path = "student"

    class DummyTeacherModel:
        def __init__(self):
            # vision_config=None — looks like a text-only model
            self.config = SimpleNamespace(vision_config=None)
            self.resized_to = None

        def resize_token_embeddings(self, vocab_size):
            self.resized_to = vocab_size

    def fake_sft_init(
        self,
        model,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=None,
        preprocess_logits_for_metrics=None,
        peft_config=None,
    ):
        del (
            data_collator,
            train_dataset,
            eval_dataset,
            compute_metrics,
            callbacks,
            optimizers,
        )
        del preprocess_logits_for_metrics, peft_config
        self.model = model
        self.args = args
        self.processing_class = processing_class
        self.accelerator = SimpleNamespace(
            device=torch.device("cpu"),
            num_processes=1,
            prepare_model=lambda module, evaluation_mode=True: module,
        )
        self.is_deepspeed_enabled = False
        self.is_fsdp_enabled = False

    monkeypatch.setattr(gold_trainer_module.SFTTrainer, "__init__", fake_sft_init)

    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")

    vision_dataset = Dataset.from_dict({"messages": [["dummy"]], "image": ["fake_image"]})

    args = SimpleNamespace(
        model_init_kwargs=None,
        max_length=128,
        truncation_mode="keep_start",
        use_liger_kernel=False,
        trust_remote_code=False,
        teacher_model_init_kwargs=None,
        use_uld_loss=False,
        teacher_tokenizer_name_or_path=None,
        teacher_model_revision=None,
        disable_dropout=False,
        lmbda=1.0,
        beta=0.5,
        temperature=1.0,
        top_p=1.0,
        seq_kd=False,
        num_generations=1,
        max_completion_length=16,
        top_k=0,
        log_completions=False,
        log_completions_steps=100,
        wandb_log_unique_prompts=True,
        num_completions_to_print=None,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        use_vllm=False,
    )

    with pytest.raises(ValueError, match="VLM distillation requires both student and teacher"):
        GOLDTrainer(
            model=DummyStudentModel(),
            teacher_model=DummyTeacherModel(),
            args=args,
            train_dataset=vision_dataset,
            processing_class=processor,
        )


def test_gold_trainer_init_rejects_keep_end_truncation_for_vlm(monkeypatch):
    """GOLDTrainer should raise ValueError when truncation_mode='keep_end' is used with a VLM."""

    class DummyStudentModel:
        def __init__(self):
            self.config = SimpleNamespace(
                _name_or_path="student", vocab_size=17, vision_config=True, model_type="dummy_vlm"
            )
            self.config.get_text_config = lambda: self.config
            self.generation_config = SimpleNamespace(eos_token_id=2)
            self.name_or_path = "student"

    class DummyTeacherModel:
        def __init__(self):
            self.config = SimpleNamespace(vision_config=True, model_type="dummy_vlm")
            self.resized_to = None

        def resize_token_embeddings(self, vocab_size):
            self.resized_to = vocab_size

    def fake_sft_init(
        self,
        model,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=None,
        preprocess_logits_for_metrics=None,
        peft_config=None,
    ):
        del data_collator, train_dataset, eval_dataset, compute_metrics, callbacks, optimizers
        del preprocess_logits_for_metrics, peft_config
        self.model = model
        self.args = args
        self.processing_class = processing_class
        self.accelerator = SimpleNamespace(
            device=torch.device("cpu"),
            num_processes=1,
            prepare_model=lambda module, evaluation_mode=True: module,
        )
        self.is_deepspeed_enabled = False
        self.is_fsdp_enabled = False

    monkeypatch.setattr(gold_trainer_module.SFTTrainer, "__init__", fake_sft_init)

    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")

    vision_dataset = Dataset.from_dict({"messages": [["dummy"]], "image": ["fake_image"]})

    args = SimpleNamespace(
        model_init_kwargs=None,
        max_length=128,
        truncation_mode="keep_end",
        use_liger_kernel=False,
        teacher_model_init_kwargs=None,
        use_uld_loss=False,
        teacher_tokenizer_name_or_path=None,
        teacher_model_revision=None,
        disable_dropout=False,
        lmbda=1.0,
        beta=0.5,
        temperature=1.0,
        top_p=1.0,
        seq_kd=False,
        num_generations=1,
        max_completion_length=16,
        top_k=0,
        log_completions=False,
        log_completions_steps=100,
        wandb_log_unique_prompts=True,
        num_completions_to_print=None,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        use_vllm=False,
    )

    with pytest.raises(ValueError, match="truncation_mode='keep_end' is not supported for vision-language models"):
        GOLDTrainer(
            model=DummyStudentModel(),
            teacher_model=DummyTeacherModel(),
            args=args,
            train_dataset=vision_dataset,
            processing_class=processor,
        )


def test_gold_trainer_vlm_vllm_init_uses_identity_collator(monkeypatch):
    """When a VLM processor is used with lmbda > 0 and use_vllm=True, GOLDTrainer should use the identity collator
    and store a _vlm_collator for on-the-fly collation. vLLM should be initialized with max_model_length from args.
    """
    captured = {}

    class DummyStudentModel:
        def __init__(self):
            self.config = SimpleNamespace(
                _name_or_path="student",
                vocab_size=17,
                vision_config=True,
                model_type="dummy_vlm",
            )
            self.config.get_text_config = lambda: self.config
            self.generation_config = SimpleNamespace(eos_token_id=2)
            self.name_or_path = "student"

    class DummyTeacherModel:
        def __init__(self):
            self.config = SimpleNamespace(vision_config=True, model_type="dummy_vlm")
            self.resized_to = None

        def resize_token_embeddings(self, vocab_size):
            self.resized_to = vocab_size

    def fake_sft_init(
        self,
        model,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=None,
        preprocess_logits_for_metrics=None,
        peft_config=None,
    ):
        self.data_collator = data_collator
        del train_dataset, eval_dataset, compute_metrics, callbacks, optimizers
        del preprocess_logits_for_metrics, peft_config
        self.model = model
        self.args = args
        self.processing_class = processing_class
        self.accelerator = SimpleNamespace(
            device=torch.device("cpu"),
            num_processes=1,
            prepare_model=lambda module, evaluation_mode=True: module,
        )
        self.is_deepspeed_enabled = False
        self.is_fsdp_enabled = False

    class CapturingVLLMGeneration:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(gold_trainer_module.SFTTrainer, "__init__", fake_sft_init)
    monkeypatch.setattr(gold_trainer_module, "is_vllm_available", lambda: True)
    monkeypatch.setattr(gold_trainer_module, "VLLMGeneration", CapturingVLLMGeneration)

    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    vision_dataset = Dataset.from_dict({"messages": [["dummy"]], "image": ["fake_image"]})

    args = SimpleNamespace(
        model_init_kwargs=None,
        max_length=128,
        truncation_mode="keep_start",
        use_liger_kernel=False,
        trust_remote_code=False,
        teacher_model_init_kwargs=None,
        use_uld_loss=False,
        teacher_tokenizer_name_or_path=None,
        teacher_model_revision=None,
        disable_dropout=False,
        lmbda=1.0,
        beta=0.5,
        temperature=1.0,
        top_p=1.0,
        seq_kd=False,
        num_generations=1,
        max_completion_length=16,
        top_k=0,
        log_completions=False,
        log_completions_steps=100,
        wandb_log_unique_prompts=True,
        num_completions_to_print=None,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_structured_outputs_regex=None,
        vllm_server_base_url=None,
        vllm_server_host="0.0.0.0",
        vllm_server_port=8001,
        vllm_group_port=51216,
        vllm_server_timeout=240.0,
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=0.2,
        vllm_max_model_length=None,
        vllm_enable_sleep_mode=False,
        vllm_model_impl="vllm",
        vllm_sync_frequency=1,
    )

    teacher_model = DummyTeacherModel()
    trainer = GOLDTrainer(
        model=DummyStudentModel(),
        teacher_model=teacher_model,
        args=args,
        train_dataset=vision_dataset,
        processing_class=processor,
    )

    # Same assertions as text-only vLLM test
    assert teacher_model.resized_to == 17
    assert captured["max_model_length"] == 128

    # VLM-specific: identity collator + _vlm_collator for on-the-fly use
    assert trainer.data_collator is identity
    assert trainer._vlm_collator is not None
    assert isinstance(trainer._vlm_collator, DataCollatorForVisionLanguageChatML)


def _make_dummy_vlm_models(student_model_type, teacher_model_type):
    """Helper to create dummy student/teacher VLM models with specified model_type."""

    class DummyStudentModel:
        def __init__(self):
            self.config = SimpleNamespace(
                _name_or_path="student",
                vocab_size=17,
                vision_config=True,
                model_type=student_model_type,
            )
            self.config.get_text_config = lambda: self.config
            self.generation_config = SimpleNamespace(eos_token_id=2)
            self.name_or_path = "student"

    class DummyTeacherModel:
        def __init__(self):
            self.config = SimpleNamespace(
                _name_or_path="teacher",
                vision_config=True,
                model_type=teacher_model_type,
            )
            self.resized_to = None

        def resize_token_embeddings(self, vocab_size):
            self.resized_to = vocab_size

    return DummyStudentModel(), DummyTeacherModel()


def _make_vlm_trainer_args(use_vllm=False):
    """Helper to create minimal GOLDTrainer args for VLM tests."""
    return SimpleNamespace(
        model_init_kwargs=None,
        max_length=128,
        truncation_mode="keep_start",
        use_liger_kernel=False,
        trust_remote_code=False,
        teacher_model_init_kwargs=None,
        use_uld_loss=False,
        teacher_tokenizer_name_or_path=None,
        teacher_model_revision=None,
        disable_dropout=False,
        lmbda=0.5,
        beta=0.5,
        temperature=1.0,
        top_p=1.0,
        seq_kd=False,
        num_generations=1,
        max_completion_length=16,
        top_k=0,
        log_completions=False,
        log_completions_steps=100,
        wandb_log_unique_prompts=True,
        num_completions_to_print=None,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        use_vllm=use_vllm,
        vllm_mode="colocate",
        vllm_structured_outputs_regex=None,
        vllm_server_base_url=None,
        vllm_server_host="0.0.0.0",
        vllm_server_port=8001,
        vllm_group_port=51216,
        vllm_server_timeout=240.0,
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=0.2,
        vllm_max_model_length=None,
        vllm_enable_sleep_mode=False,
        vllm_model_impl="vllm",
        vllm_sync_frequency=1,
        # ULD-specific defaults (needed when use_uld_loss=True)
        uld_crossentropy_weight=0.5,
        uld_distillation_weight=0.5,
        uld_student_temperature=1.0,
        uld_teacher_temperature=1.0,
        uld_skip_student_eos=False,
        uld_skip_teacher_eos=False,
        use_extended_uld=False,
        uld_token_merge_strategy="observed",
    )


def test_cross_architecture_vlm_without_uld_raises_error(monkeypatch):
    """When student and teacher have different model_type and use_uld_loss=False, GOLDTrainer should raise
    a ValueError telling the user to enable ULD loss."""

    def fake_sft_init(
        self,
        model,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=None,
        preprocess_logits_for_metrics=None,
        peft_config=None,
    ):
        self.data_collator = data_collator
        self.model = model
        self.args = args
        self.processing_class = processing_class
        self.accelerator = SimpleNamespace(
            device=torch.device("cpu"),
            num_processes=1,
            prepare_model=lambda module, evaluation_mode=True: module,
        )
        self.is_deepspeed_enabled = False
        self.is_fsdp_enabled = False

    monkeypatch.setattr(gold_trainer_module.SFTTrainer, "__init__", fake_sft_init)

    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    sentinel_processor = SimpleNamespace(_is_sentinel=True)
    real_auto_processor_from_pretrained = AutoProcessor.from_pretrained

    def patched_auto_processor(name, **kwargs):
        if name == "teacher":
            return sentinel_processor
        return real_auto_processor_from_pretrained(name, **kwargs)

    monkeypatch.setattr(
        gold_trainer_module.AutoProcessor,
        "from_pretrained",
        staticmethod(patched_auto_processor),
    )

    vision_dataset = Dataset.from_dict({"messages": [["dummy"]], "image": ["fake_image"]})
    student, teacher = _make_dummy_vlm_models("smolvlm", "qwen2_5_vl")
    args = _make_vlm_trainer_args()  # use_uld_loss=False by default

    with pytest.raises(ValueError, match="Cross-architecture VLM distillation.*use_uld_loss=True"):
        GOLDTrainer(
            model=student,
            teacher_model=teacher,
            args=args,
            train_dataset=vision_dataset,
            processing_class=processor,
        )


def test_cross_architecture_vlm_with_uld_sets_teacher_processor(monkeypatch):
    """When student and teacher have different model_type and use_uld_loss=True, GOLDTrainer should store
    a separate _teacher_processor and emit a warning."""

    def fake_sft_init(
        self,
        model,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=None,
        preprocess_logits_for_metrics=None,
        peft_config=None,
    ):
        self.data_collator = data_collator
        self.model = model
        self.args = args
        self.processing_class = processing_class
        self.accelerator = SimpleNamespace(
            device=torch.device("cpu"),
            num_processes=1,
            prepare_model=lambda module, evaluation_mode=True: module,
        )
        self.is_deepspeed_enabled = False
        self.is_fsdp_enabled = False

    monkeypatch.setattr(gold_trainer_module.SFTTrainer, "__init__", fake_sft_init)

    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    sentinel_processor = SimpleNamespace(_is_sentinel=True)
    real_auto_processor_from_pretrained = AutoProcessor.from_pretrained

    def patched_auto_processor(name, **kwargs):
        if name == "teacher":
            return sentinel_processor
        return real_auto_processor_from_pretrained(name, **kwargs)

    monkeypatch.setattr(
        gold_trainer_module.AutoProcessor,
        "from_pretrained",
        staticmethod(patched_auto_processor),
    )

    # Monkeypatch AutoTokenizer.from_pretrained for ULD teacher tokenizer loading
    sentinel_tokenizer = SimpleNamespace(pad_token="<pad>", eos_token="</s>")
    sentinel_processor.tokenizer = sentinel_tokenizer
    real_auto_tokenizer_from_pretrained = AutoTokenizer.from_pretrained

    def patched_auto_tokenizer(name, **kwargs):
        if name == "teacher":
            return sentinel_tokenizer
        return real_auto_tokenizer_from_pretrained(name, **kwargs)

    monkeypatch.setattr(
        gold_trainer_module.AutoTokenizer,
        "from_pretrained",
        staticmethod(patched_auto_tokenizer),
    )

    vision_dataset = Dataset.from_dict({"messages": [["dummy"]], "image": ["fake_image"]})
    student, teacher = _make_dummy_vlm_models("smolvlm", "qwen2_5_vl")
    args = _make_vlm_trainer_args()
    args.use_uld_loss = True
    args.teacher_tokenizer_name_or_path = "teacher"

    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trainer = GOLDTrainer(
            model=student,
            teacher_model=teacher,
            args=args,
            train_dataset=vision_dataset,
            processing_class=processor,
        )

    # _teacher_processor should be set for cross-architecture
    assert trainer._teacher_processor is not None
    assert trainer._teacher_processor is sentinel_processor
    assert trainer._is_cross_architecture_vlm is True

    # A cross-architecture warning should have been emitted
    cross_arch_warnings = [w for w in caught if "Cross-architecture VLM distillation" in str(w.message)]
    assert len(cross_arch_warnings) == 1
    assert "smolvlm" in str(cross_arch_warnings[0].message)
    assert "qwen2_5_vl" in str(cross_arch_warnings[0].message)

    # Identity collator and VLM collator should still be set
    assert trainer.data_collator is identity
    assert trainer._vlm_collator is not None


def test_same_architecture_vlm_no_teacher_processor(monkeypatch):
    """When student and teacher have the same model_type, GOLDTrainer should NOT store a _teacher_processor
    (zero overhead -- both models share the same forward_kwargs)."""

    def fake_sft_init(
        self,
        model,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=None,
        preprocess_logits_for_metrics=None,
        peft_config=None,
    ):
        self.data_collator = data_collator
        self.model = model
        self.args = args
        self.processing_class = processing_class
        self.accelerator = SimpleNamespace(
            device=torch.device("cpu"),
            num_processes=1,
            prepare_model=lambda module, evaluation_mode=True: module,
        )
        self.is_deepspeed_enabled = False
        self.is_fsdp_enabled = False

    monkeypatch.setattr(gold_trainer_module.SFTTrainer, "__init__", fake_sft_init)

    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    vision_dataset = Dataset.from_dict({"messages": [["dummy"]], "image": ["fake_image"]})
    student, teacher = _make_dummy_vlm_models("smolvlm", "smolvlm")
    args = _make_vlm_trainer_args()

    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trainer = GOLDTrainer(
            model=student,
            teacher_model=teacher,
            args=args,
            train_dataset=vision_dataset,
            processing_class=processor,
        )

    # _teacher_processor should be None for same architecture (zero overhead)
    assert trainer._teacher_processor is None
    assert trainer._is_cross_architecture_vlm is False

    # No cross-architecture warning should have been emitted
    cross_arch_warnings = [w for w in caught if "Cross-architecture VLM distillation" in str(w.message)]
    assert len(cross_arch_warnings) == 0

    # Identity collator and VLM collator should still be set
    assert trainer.data_collator is identity
    assert trainer._vlm_collator is not None


def test_same_architecture_vlm_with_uld_sets_teacher_processor(monkeypatch):
    """ULD VLM distillation should use a teacher processor even when the VLM model_type matches."""

    def fake_sft_init(
        self,
        model,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=None,
        preprocess_logits_for_metrics=None,
        peft_config=None,
    ):
        self.data_collator = data_collator
        self.model = model
        self.args = args
        self.processing_class = processing_class
        self.accelerator = SimpleNamespace(
            device=torch.device("cpu"),
            num_processes=1,
            prepare_model=lambda module, evaluation_mode=True: module,
        )
        self.is_deepspeed_enabled = False
        self.is_fsdp_enabled = False

    monkeypatch.setattr(gold_trainer_module.SFTTrainer, "__init__", fake_sft_init)

    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    sentinel_processor = SimpleNamespace(_is_sentinel=True)
    real_auto_processor_from_pretrained = AutoProcessor.from_pretrained

    def patched_auto_processor(name, **kwargs):
        if name == "teacher":
            return sentinel_processor
        return real_auto_processor_from_pretrained(name, **kwargs)

    monkeypatch.setattr(
        gold_trainer_module.AutoProcessor,
        "from_pretrained",
        staticmethod(patched_auto_processor),
    )

    sentinel_tokenizer = SimpleNamespace(pad_token="<pad>", eos_token="</s>")
    sentinel_processor.tokenizer = sentinel_tokenizer
    real_auto_tokenizer_from_pretrained = AutoTokenizer.from_pretrained

    def patched_auto_tokenizer(name, **kwargs):
        if name == "teacher":
            return sentinel_tokenizer
        return real_auto_tokenizer_from_pretrained(name, **kwargs)

    monkeypatch.setattr(
        gold_trainer_module.AutoTokenizer,
        "from_pretrained",
        staticmethod(patched_auto_tokenizer),
    )

    vision_dataset = Dataset.from_dict({"messages": [["dummy"]], "image": ["fake_image"]})
    student, teacher = _make_dummy_vlm_models("smolvlm", "smolvlm")
    args = _make_vlm_trainer_args()
    args.use_uld_loss = True
    args.teacher_tokenizer_name_or_path = "teacher"

    trainer = GOLDTrainer(
        model=student,
        teacher_model=teacher,
        args=args,
        train_dataset=vision_dataset,
        processing_class=processor,
    )

    assert trainer._teacher_processor is sentinel_processor
    assert trainer._is_cross_architecture_vlm is False
    assert trainer.teacher_tokenizer is sentinel_tokenizer
    assert trainer.data_collator is identity
    assert trainer._vlm_collator is not None


def test_same_architecture_vlm_uld_preserves_raw_images_for_teacher_processor(
    monkeypatch,
):
    trainer = GOLDTrainer.__new__(GOLDTrainer)
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"), is_main_process=True)
    trainer.args = SimpleNamespace(gradient_accumulation_steps=2)
    trainer.lmbda = 0.0
    trainer.use_uld_loss = True
    trainer.teacher_tokenizer = SimpleNamespace(pad_token_id=0)
    trainer._teacher_processor = object()
    trainer._is_cross_architecture_vlm = False
    trainer._step = 0
    trainer.model = SimpleNamespace(training=True)

    def stub_collator(examples):
        return {
            "input_ids": torch.zeros(len(examples), 1, dtype=torch.long),
            "original_prompt_text": [example["prompt"][0]["content"] for example in examples],
            "original_completion_text": [example["completion"][0]["content"] for example in examples],
        }

    trainer._vlm_collator = stub_collator
    monkeypatch.setattr(
        gold_trainer_module,
        "broadcast_object_list",
        lambda values, from_process: values,
    )
    monkeypatch.setattr(
        gold_trainer_module,
        "prepare_multimodal_messages",
        lambda prompt, images: prompt,
    )

    images = [object(), object()]
    generation_batch = [
        {
            "prompt": [{"role": "user", "content": "q0"}],
            "completion": [{"role": "assistant", "content": "a0"}],
            "image": images[0],
        },
        {
            "prompt": [{"role": "user", "content": "q1"}],
            "completion": [{"role": "assistant", "content": "a1"}],
            "image": images[1],
        },
    ]

    first_slice = trainer._prepare_inputs(generation_batch)

    assert first_slice["_raw_images"] == [[images[0]]]
    assert first_slice["_raw_prompts"] == [generation_batch[0]["prompt"]]
    assert "_gold_vlm_raw_images" in trainer._buffered_inputs[1]
    assert "_gold_vlm_raw_prompts" in trainer._buffered_inputs[1]


def test_on_policy_vlm_vllm_does_not_duplicate_repeated_sampler_batch(monkeypatch):
    """The VLM vLLM path must rely on RepeatSampler for `num_generations` duplication.

    `VLLMGeneration.generate` expects the incoming prompt batch to already contain the repeated prompt entries,
    matching the text-only path. Duplicating here again would produce `num_generations ** 2` completions.
    """
    num_generations = 3
    num_slices = 2

    trainer = GOLDTrainer.__new__(GOLDTrainer)
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
    trainer.use_vllm = True
    trainer.num_generations = num_generations
    trainer.state = SimpleNamespace(global_step=0)
    trainer._last_vllm_sync_step = -1
    trainer.vllm_sync_frequency = 1
    trainer.generation_config = SimpleNamespace(max_new_tokens=16)
    trainer.args = SimpleNamespace(max_length=32)  # budget of 32 - 16 = 16 fits the 5-token stub prompts
    trainer._buffered_inputs = {}
    trainer._buffered_text_logs = {}
    trainer._teacher_processor = None
    trainer._is_cross_architecture_vlm = False
    trainer.use_uld_loss = False
    trainer.teacher_tokenizer = None

    class StubProcessor:
        @staticmethod
        def apply_chat_template(conversation, add_generation_prompt, tokenize, return_dict, **kwargs):
            return {
                "input_ids": [[1, 2, 3, 4, 5] for _ in conversation],
                "attention_mask": [[1, 1, 1, 1, 1] for _ in conversation],
            }

        @staticmethod
        def batch_decode(ids, skip_special_tokens):
            return [f"prompt_{i}" for i in range(len(ids))]

        @staticmethod
        def decode(ids, skip_special_tokens, clean_up_tokenization_spaces):
            tokens = []
            for token_id in ids:
                if token_id == 9:
                    if skip_special_tokens:
                        continue
                    tokens.append("<eos>")
                else:
                    tokens.append(f"comp_{token_id}")
            return "".join(tokens)

    trainer.processing_class = StubProcessor

    received = {}

    class StubVLLMGeneration:
        def sync_weights(self):
            pass

        def generate(self, prompts, images, num_generations):
            received["n_prompts"] = len(prompts)
            received["n_images"] = len(images) if images is not None else None
            received["prompts"] = prompts
            completion_ids = [[100 + i, 9] for i in range(len(prompts))]
            return None, completion_ids, None, None

    trainer.vllm_generation = StubVLLMGeneration()

    collated_per_call = []

    def stub_collator(synthetic_examples):
        collated_per_call.append(list(synthetic_examples))
        return {
            "input_ids": torch.zeros(len(synthetic_examples), 1, dtype=torch.long),
            "original_prompt_text": [example["prompt"][0]["content"] for example in synthetic_examples],
            "original_completion_text": [
                example["completion"][0]["content"][0]["text"] for example in synthetic_examples
            ],
        }

    trainer._vlm_collator = stub_collator

    class FakeImage:
        def __init__(self, tag):
            self.tag = tag

    unique_prompts_per_slice = 2
    unique_examples = [
        {"prompt": [{"role": "user", "content": f"q{i}"}], "image": FakeImage(str(i))}
        for i in range(num_slices * unique_prompts_per_slice)
    ]
    sampler = RepeatSampler(
        unique_examples,
        mini_repeat_count=num_generations,
        batch_size=len(unique_examples),
        shuffle=False,
    )
    sampled_examples = [unique_examples[i] for i in sampler]
    raw_slices = [
        sampled_examples[i : i + unique_prompts_per_slice * num_generations]
        for i in range(0, len(sampled_examples), unique_prompts_per_slice * num_generations)
    ]
    on_policy_indices = list(range(num_slices))

    # Bypass multimodal-message helper; its exact shape is irrelevant to this regression.
    monkeypatch.setattr(
        gold_trainer_module,
        "prepare_multimodal_messages",
        lambda prompt, images: prompt,
    )

    trainer._generate_on_policy_vlm_raw(raw_slices, on_policy_indices)

    total_sampled_prompts = num_slices * unique_prompts_per_slice * num_generations
    assert received["n_prompts"] == total_sampled_prompts
    assert received["n_images"] == total_sampled_prompts
    assert all(prompt == [1, 2, 3, 4, 5] for prompt in received["prompts"])

    # Synthetic VLM examples are stored lazily and are not collated until their slice is consumed.
    assert len(collated_per_call) == 0

    # Buffers populated for every on-policy slice without IndexError.
    for slice_idx in on_policy_indices:
        assert slice_idx in trainer._buffered_inputs
        _, completion_texts = trainer._buffered_text_logs[slice_idx]
        assert len(completion_texts) == unique_prompts_per_slice * num_generations

    first_slice = trainer._materialize_vlm_slice(trainer._buffered_inputs[0])
    assert first_slice["input_ids"].shape[0] == unique_prompts_per_slice * num_generations
    assert first_slice["original_prompt_text"] == [example["prompt"][0]["content"] for example in collated_per_call[0]]
    assert all(completion.endswith("<eos>") for completion in first_slice["original_completion_text"])
    assert all("<" not in prompt for prompt in first_slice["original_prompt_text"])
    assert len(collated_per_call) == 1
    assert len(collated_per_call[0]) == unique_prompts_per_slice * num_generations


def test_vlm_uld_custom_collator_missing_raw_fields_raises_clear_error():
    trainer = GOLDTrainer.__new__(GOLDTrainer)
    trainer.use_uld_loss = True
    trainer.teacher_tokenizer = object()
    trainer._teacher_processor = object()

    inputs = {
        "input_ids": torch.ones(1, 2, dtype=torch.long),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
        "original_prompt_text": ["prompt"],
        "original_completion_text": ["completion"],
    }

    with pytest.raises(ValueError, match="requires `_raw_images` and `_raw_prompts`"):
        GOLDTrainer.compute_loss(trainer, model=object(), inputs=inputs)


def test_off_policy_vlm_collates_only_consumed_slice(monkeypatch):
    trainer = GOLDTrainer.__new__(GOLDTrainer)
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"), is_main_process=True)
    trainer.args = SimpleNamespace(gradient_accumulation_steps=2)
    trainer.lmbda = 0.0
    trainer.use_uld_loss = False
    trainer.teacher_tokenizer = None
    trainer._teacher_processor = None
    trainer._is_cross_architecture_vlm = False
    trainer._step = 0
    trainer.model = SimpleNamespace(training=True)
    collated_per_call = []

    def stub_collator(examples):
        collated_per_call.append(list(examples))
        return {"input_ids": torch.zeros(len(examples), 1, dtype=torch.long)}

    trainer._vlm_collator = stub_collator
    monkeypatch.setattr(
        gold_trainer_module,
        "broadcast_object_list",
        lambda values, from_process: values,
    )

    generation_batch = [
        {"prompt": [{"role": "user", "content": "q0"}], "image": object()},
        {"prompt": [{"role": "user", "content": "q1"}], "image": object()},
        {"prompt": [{"role": "user", "content": "q2"}], "image": object()},
        {"prompt": [{"role": "user", "content": "q3"}], "image": object()},
    ]

    first_slice = trainer._prepare_inputs(generation_batch)

    assert len(collated_per_call) == 1
    assert first_slice["input_ids"].shape[0] == 2
    assert "_gold_vlm_lazy_examples" in trainer._buffered_inputs[1]

    second_slice = trainer._prepare_inputs(generation_batch)

    assert len(collated_per_call) == 2
    assert second_slice["input_ids"].shape[0] == 2


def test_eval_vlm_collates_raw_batch_off_policy():
    """VLM eval (identity collator yields raw dicts) must collate off-policy in `_prepare_inputs`.

    Regression test for the eval crash: the inherited path indexed the raw `list[dict]`. Eval must run the VLM collator
    over the whole batch (no slicing, no buffering, no generation).
    """
    trainer = GOLDTrainer.__new__(GOLDTrainer)
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"), is_main_process=True)
    trainer.model = SimpleNamespace(training=False)
    trainer.use_uld_loss = False
    trainer.teacher_tokenizer = None
    trainer._teacher_processor = None
    collated_per_call = []

    def stub_collator(examples):
        collated_per_call.append(list(examples))
        return {"input_ids": torch.zeros(len(examples), 1, dtype=torch.long)}

    trainer._vlm_collator = stub_collator

    generation_batch = [
        {"prompt": [{"role": "user", "content": "q0"}], "image": object()},
        {"prompt": [{"role": "user", "content": "q1"}], "image": object()},
        {"prompt": [{"role": "user", "content": "q2"}], "image": object()},
    ]

    inputs = trainer._prepare_inputs(generation_batch)

    # The whole eval batch is collated once (no per-accumulation-step slicing) into a tensor dict.
    assert len(collated_per_call) == 1
    assert collated_per_call[0] == generation_batch
    assert inputs["input_ids"].shape[0] == len(generation_batch)
    # Off-policy only: no generation occurred, so no on-policy buffer state was created.
    assert not hasattr(trainer, "_buffered_inputs") or trainer._buffered_inputs is None


def test_eval_vlm_attaches_raw_images_for_teacher_processor(monkeypatch):
    """When a teacher processor is configured (cross-arch / ULD), eval must attach raw images and prompts."""
    trainer = GOLDTrainer.__new__(GOLDTrainer)
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"), is_main_process=True)
    trainer.model = SimpleNamespace(training=False)
    trainer.use_uld_loss = False
    trainer.teacher_tokenizer = None
    trainer._teacher_processor = object()

    def stub_collator(examples):
        return {"input_ids": torch.zeros(len(examples), 1, dtype=torch.long)}

    trainer._vlm_collator = stub_collator
    # The exact multimodal-message shape is irrelevant here; pass the prompt through unchanged.
    monkeypatch.setattr(
        gold_trainer_module,
        "prepare_multimodal_messages",
        lambda prompt, images: prompt,
    )

    img0, img1 = object(), object()
    generation_batch = [
        {"prompt": [{"role": "user", "content": "q0"}], "image": img0},
        {"prompt": [{"role": "user", "content": "q1"}], "image": img1},
    ]

    inputs = trainer._prepare_inputs(generation_batch)

    assert inputs["_raw_images"] == [[img0], [img1]]
    assert inputs["_raw_prompts"] == [ex["prompt"] for ex in generation_batch]


def test_on_policy_vlm_without_vllm_collates_only_consumed_slice(monkeypatch):
    trainer = GOLDTrainer.__new__(GOLDTrainer)
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"), is_main_process=True)
    trainer.args = SimpleNamespace(gradient_accumulation_steps=2, max_length=None)
    trainer.use_vllm = False
    trainer._teacher_processor = None
    trainer._is_cross_architecture_vlm = False
    trainer._buffered_inputs = [None, None]
    trainer._buffered_text_logs = [None, None]
    trainer._step = 1
    trainer.generation_kwargs = {}
    trainer.generation_config = SimpleNamespace(max_new_tokens=1)
    trainer.pad_token_id = 0
    trainer.use_uld_loss = False
    trainer.teacher_tokenizer = None
    trainer.uld_loss_fn = None
    collated_per_call = []

    class StubProcessor:
        @staticmethod
        def apply_chat_template(conversation, add_generation_prompt, tokenize, return_dict, padding):
            return {
                "input_ids": [[1, 2] for _ in conversation],
                "attention_mask": [[1, 1] for _ in conversation],
            }

        @staticmethod
        def batch_decode(ids, skip_special_tokens):
            return [f"prompt_{i}" for i in range(len(ids))]

        @staticmethod
        def decode(ids, skip_special_tokens, clean_up_tokenization_spaces):
            tokens = []
            for token_id in ids:
                if token_id == 9:
                    if skip_special_tokens:
                        continue
                    tokens.append("<eos>")
                else:
                    tokens.append(f"tok{token_id}")
            return "".join(tokens)

    trainer.processing_class = StubProcessor

    def stub_collator(examples):
        collated_per_call.append(list(examples))
        assert all(example.get("completion") == "" for example in examples)
        batch_size = len(examples)
        return {
            "prompts": torch.ones(batch_size, 2, dtype=torch.long),
            "prompt_attention_mask": torch.ones(batch_size, 2, dtype=torch.long),
            "pixel_values": torch.zeros(batch_size, 3, 2, 2),
            "spatial_shapes": torch.tensor([[2, 2]] * batch_size, dtype=torch.long),
            "original_prompt_text": [example["prompt"][0]["content"] for example in examples],
        }

    trainer._vlm_collator = stub_collator

    class FakeModel:
        training = True

        @staticmethod
        def generate(
            input_ids,
            attention_mask,
            generation_config,
            return_dict_in_generate,
            **kwargs,
        ):
            assert "spatial_shapes" in kwargs
            assert torch.equal(kwargs["spatial_shapes"], torch.tensor([[2, 2]], dtype=torch.long))
            completion = torch.tensor([[3, 9]] * input_ids.shape[0], dtype=torch.long)
            return SimpleNamespace(sequences=torch.cat([input_ids, completion], dim=1))

    trainer.model = FakeModel()

    monkeypatch.setattr(
        gold_trainer_module,
        "unwrap_model_for_generation",
        lambda *args, **kwargs: gold_trainer_module.nullcontext(args[0]),
    )
    monkeypatch.setattr(
        gold_trainer_module,
        "prepare_multimodal_messages",
        lambda prompt, images: prompt,
    )

    raw_slices = [
        [
            {
                "prompt": [{"role": "user", "content": "q0"}],
                "completion": "gold0",
                "image": object(),
            }
        ],
        [
            {
                "prompt": [{"role": "user", "content": "q1"}],
                "completion": "gold1",
                "image": object(),
            }
        ],
    ]

    trainer._generate_on_policy_vlm_raw(raw_slices, [0, 1])

    assert len(collated_per_call) == 0
    assert "_gold_vlm_on_policy_raw_examples" in trainer._buffered_inputs[0]
    assert "_gold_vlm_on_policy_raw_examples" in trainer._buffered_inputs[1]

    consumed_slice = trainer._prepare_inputs(raw_slices)

    assert len(collated_per_call) == 1
    assert consumed_slice["input_ids"].shape == (1, 4)
    assert torch.equal(consumed_slice["spatial_shapes"], torch.tensor([[2, 2]], dtype=torch.long))
    assert consumed_slice["original_prompt_text"] == ["q1"]
    # Special tokens (e.g. EOS) are kept so the text matches the supervised tokens that `byte_offsets`/ULD align on.
    assert consumed_slice["original_completion_text"] == ["tok3<eos>"]
    assert "_gold_vlm_on_policy_raw_examples" in trainer._buffered_inputs[0]
    assert "_gold_vlm_on_policy_raw_examples" not in trainer._buffered_inputs[1]


def test_model_forward_kwargs_preserve_processor_tensor_fields():
    trainer = GOLDTrainer.__new__(GOLDTrainer)
    inputs = {
        "input_ids": torch.ones(1, 2, dtype=torch.long),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
        "labels": torch.ones(1, 2, dtype=torch.long),
        "prompts": torch.ones(1, 1, dtype=torch.long),
        "prompt_attention_mask": torch.ones(1, 1, dtype=torch.long),
        "completion_mask": torch.ones(1, 2, dtype=torch.long),
        "assistant_masks": torch.ones(1, 2, dtype=torch.long),
        "original_prompt_text": ["prompt"],
        "_raw_images": [object()],
        "pixel_values": torch.zeros(1, 3, 2, 2),
        "spatial_shapes": torch.tensor([[2, 2]], dtype=torch.long),
        "custom_processor_tensor": torch.tensor([1]),
        "token_type_ids": torch.zeros(1, 2, dtype=torch.long),
    }

    kwargs = trainer._get_model_forward_kwargs(inputs)

    assert set(kwargs) == {
        "pixel_values",
        "spatial_shapes",
        "custom_processor_tensor",
        "token_type_ids",
    }
    assert set(trainer._get_model_forward_kwargs(inputs, exclude=("token_type_ids",))) == {
        "pixel_values",
        "spatial_shapes",
        "custom_processor_tensor",
    }


# End-to-end smoke tests: load tiny real VLMs from trl-internal-testing and run a single
# off-policy training step (use_vllm=False, lmbda=0.0 → deterministic gold-completion path).

_TINY_QWEN3_VL = "trl-internal-testing/tiny-Qwen3VLForConditionalGeneration"
_TINY_SMOLVLM = "trl-internal-testing/tiny-SmolVLMForConditionalGeneration"
_VLM_SMOKE_MAX_LENGTH = 4096


@pytest.mark.slow
def test_vlm_jsd_same_family_train_step_smoke(tmp_path, vlm_dataset):
    """Same-family VLM (tiny Qwen3-VL → tiny Qwen3-VL) runs one off-policy JSD step with a finite loss."""
    try:
        student = AutoModelForImageTextToText.from_pretrained(_TINY_QWEN3_VL, dtype=torch.bfloat16)
        teacher = AutoModelForImageTextToText.from_pretrained(_TINY_QWEN3_VL, dtype=torch.bfloat16)
        processor = AutoProcessor.from_pretrained(_TINY_QWEN3_VL)
    except Exception as exc:  # pragma: no cover - network/environment dependent
        pytest.skip(f"tiny Qwen3-VL assets unavailable: {exc}")
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    args = GOLDConfig(
        output_dir=str(tmp_path),
        report_to="none",
        bf16=True,
        max_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_completion_length=8,
        max_length=_VLM_SMOKE_MAX_LENGTH,
        lmbda=0.0,
        beta=0.5,
        temperature=1.0,
        num_generations=1,
        use_vllm=False,
        use_uld_loss=False,
        log_completions=False,
        save_strategy="no",
        eval_strategy="no",
        logging_strategy="no",
        dataloader_drop_last=True,
    )

    trainer = GOLDTrainer(
        model=student,
        teacher_model=teacher,
        args=args,
        train_dataset=vlm_dataset,
        processing_class=processor,
    )
    train_output = trainer.train()
    assert torch.isfinite(torch.tensor(train_output.training_loss))


_TINY_LLAMA = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"


@pytest.mark.slow
@require_liger_kernel
def test_jsd_liger_text_train_step_smoke(tmp_path):
    """Text same-family (tiny Llama → tiny Llama) runs one off-policy JSD step with the fused Liger loss.

    Exercises the `LigerFusedLinearJSDLoss` path end-to-end (`_liger_backbone` student + teacher forwards, fused
    lm_head matmul) and asserts the resulting training loss is finite.
    """
    from datasets import load_dataset

    try:
        student = AutoModelForCausalLM.from_pretrained(_TINY_LLAMA, dtype=torch.bfloat16)
        teacher = AutoModelForCausalLM.from_pretrained(_TINY_LLAMA, dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(_TINY_LLAMA)
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train[:3]")
    except Exception as exc:  # pragma: no cover - network/environment dependent
        pytest.skip(f"tiny Llama / zen assets unavailable: {exc}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    args = GOLDConfig(
        output_dir=str(tmp_path),
        report_to="none",
        bf16=True,
        max_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_completion_length=8,
        max_length=512,
        lmbda=0.0,
        beta=0.5,
        temperature=1.0,
        num_generations=1,
        use_vllm=False,
        use_uld_loss=False,
        use_liger_kernel=True,
        log_completions=False,
        save_strategy="no",
        eval_strategy="no",
        logging_strategy="no",
        dataloader_drop_last=True,
    )

    trainer = GOLDTrainer(
        model=student,
        teacher_model=teacher,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    train_output = trainer.train()
    assert torch.isfinite(torch.tensor(train_output.training_loss))


@pytest.mark.slow
@require_liger_kernel
def test_vlm_jsd_liger_same_family_train_step_smoke(tmp_path, vlm_dataset):
    """Same-family VLM (tiny Qwen3-VL → tiny Qwen3-VL) runs one off-policy JSD step with the fused Liger loss.

    Proves the VLM Liger path: `_liger_backbone` routes through `base_model` (so image features are injected) for both
    student and teacher, image kwargs reach the backbone forwards, and the fused JSD loss is finite.
    """
    try:
        student = AutoModelForImageTextToText.from_pretrained(_TINY_QWEN3_VL, dtype=torch.bfloat16)
        teacher = AutoModelForImageTextToText.from_pretrained(_TINY_QWEN3_VL, dtype=torch.bfloat16)
        processor = AutoProcessor.from_pretrained(_TINY_QWEN3_VL)
    except Exception as exc:  # pragma: no cover - network/environment dependent
        pytest.skip(f"tiny Qwen3-VL assets unavailable: {exc}")
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    args = GOLDConfig(
        output_dir=str(tmp_path),
        report_to="none",
        bf16=True,
        max_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_completion_length=8,
        max_length=_VLM_SMOKE_MAX_LENGTH,
        lmbda=0.0,
        beta=0.5,
        temperature=1.0,
        num_generations=1,
        use_vllm=False,
        use_uld_loss=False,
        use_liger_kernel=True,
        log_completions=False,
        save_strategy="no",
        eval_strategy="no",
        logging_strategy="no",
        dataloader_drop_last=True,
    )

    trainer = GOLDTrainer(
        model=student,
        teacher_model=teacher,
        args=args,
        train_dataset=vlm_dataset,
        processing_class=processor,
    )
    train_output = trainer.train()
    assert torch.isfinite(torch.tensor(train_output.training_loss))


@pytest.mark.slow
def test_vlm_uld_cross_arch_train_step_smoke(tmp_path, vlm_dataset):
    """Cross-arch VLM (tiny SmolVLM student → tiny Qwen3-VL teacher) runs one off-policy ULD step.

    Exercises `_build_teacher_vlm_inputs` (teacher rendered through its own processor), the separate teacher image
    forward, and byte-offset ULD alignment on real logits, ending in a finite loss.
    """
    try:
        student = AutoModelForImageTextToText.from_pretrained(_TINY_SMOLVLM, dtype=torch.bfloat16)
        teacher = AutoModelForImageTextToText.from_pretrained(_TINY_QWEN3_VL, dtype=torch.bfloat16)
        processor = AutoProcessor.from_pretrained(_TINY_SMOLVLM)
    except Exception as exc:  # pragma: no cover - network/environment dependent
        pytest.skip(f"tiny VLM assets unavailable: {exc}")
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    args = GOLDConfig(
        output_dir=str(tmp_path),
        report_to="none",
        bf16=True,
        max_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_completion_length=8,
        max_length=_VLM_SMOKE_MAX_LENGTH,
        lmbda=0.0,
        beta=0.5,
        temperature=1.0,
        num_generations=1,
        use_vllm=False,
        use_uld_loss=True,
        uld_crossentropy_weight=0.5,
        uld_distillation_weight=0.5,
        log_completions=False,
        save_strategy="no",
        eval_strategy="no",
        logging_strategy="no",
        dataloader_drop_last=True,
    )

    trainer = GOLDTrainer(
        model=student,
        teacher_model=teacher,
        args=args,
        train_dataset=vlm_dataset,
        processing_class=processor,
    )
    train_output = trainer.train()
    assert torch.isfinite(torch.tensor(train_output.training_loss))


class TestGOLDTrainerLoss(TrlTestCase):
    def setup_method(self):
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def test_loss_normalizes_by_num_items_in_batch(self):
        # When `num_items_in_batch` is passed (as under gradient accumulation), the JSD loss must be reduced as
        # sum / num_items_in_batch rather than the local per-microbatch mean. The batch uses variable-length prompts
        # to ensure the loss covers every valid completion token instead of slicing by the batch-max prompt width.
        # See issue #4719. The ULD path has its own normalization and is not covered here.
        dataset = Dataset.from_dict(
            {
                "messages": [
                    [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello there, how are you?"}],
                    [
                        {"role": "user", "content": "Please explain in detail the theory of general relativity"},
                        {"role": "assistant", "content": "OK"},
                    ],
                ]
            }
        )
        trainer = GOLDTrainer(
            model=self.model_id,
            teacher_model=self.model_id,
            args=GOLDConfig(
                output_dir=self.tmp_dir,
                report_to="none",
                per_device_train_batch_size=2,
                max_length=64,
                max_completion_length=20,
                use_cpu=True,
                bf16=False,
            ),
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )

        # Diverge the teacher from the student so JSD is well above fp noise (else the loss is identically 0).
        torch.manual_seed(0)
        with torch.no_grad():
            for p in trainer.teacher_model.parameters():
                p.add_(0.5 * torch.randn_like(p))

        device = next(trainer.model.parameters()).device
        batch = trainer.data_collator([trainer.train_dataset[i] for i in range(2)])
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        prompt_width = batch["prompts"].shape[1]
        old_prompt_width_count = (batch["labels"][:, prompt_width:] != -100).sum()
        num_valid = (batch["labels"] != -100).sum()

        # Prove this batch exposes the regression: the old prompt-width slice would miss valid completion labels.
        assert prompt_width > (batch["labels"][0] != -100).nonzero()[0].item()
        assert num_valid > old_prompt_width_count

        trainer.model.eval()
        with torch.no_grad():
            loss_mean = trainer.compute_loss(trainer.model, batch)  # num_items_in_batch=None -> local mean
            loss_global = trainer.compute_loss(trainer.model, batch, num_items_in_batch=num_valid)

        # With num_items_in_batch equal to the local valid-token count, sum/N equals the local mean.
        torch.testing.assert_close(loss_global, loss_mean, rtol=1e-4, atol=1e-6)

        # Doubling the global count exactly halves the loss (sum / num_items is linear in 1/num_items).
        loss_double = trainer.compute_loss(trainer.model, batch, num_items_in_batch=num_valid * 2)
        torch.testing.assert_close(loss_double, loss_mean / 2, rtol=1e-4, atol=1e-6)
