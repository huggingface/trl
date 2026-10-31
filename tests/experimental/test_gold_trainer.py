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
import warnings
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import transformers
from datasets import Dataset, DatasetDict, IterableDatasetDict, load_dataset
from packaging.version import Version
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
)

from trl.data_utils import prepare_multimodal_messages
from trl.experimental.gold import GOLDConfig
from trl.experimental.gold import gold_trainer as gold_trainer_module
from trl.experimental.gold.gold_trainer import (
    GOLDTrainer,
    ULDLoss,
    build_teacher_inputs,
)
from trl.experimental.utils import (
    DataCollatorForChatML,
    DataCollatorForVisionLanguageChatML,
)
from trl.trainer.utils import RepeatSampler, identity

from ..testing_utils import TrlTestCase


_TINY_QWEN3_VL = "trl-internal-testing/tiny-Qwen3VLForConditionalGeneration"
_TINY_SMOLVLM = "trl-internal-testing/tiny-SmolVLMForConditionalGeneration"
_TINY_LFM2_VL = "trl-internal-testing/tiny-Lfm2VlForConditionalGeneration-2.5"
_TINY_LLAMA = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"
_TINY_QWEN2 = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
_TINY_QWEN2_REVISION = "8913f5819566"


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
    """Build teacher inputs the way `GOLDTrainer.compute_loss` does: the teacher continues from the text of the
    student's completion, up to its first special token (its end-of-turn)."""
    special_ids = {token_id for token_id, token in student_tok.added_tokens_decoder.items() if token.special}
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

        completion_ids = input_ids_tensor[labels_tensor != -100].tolist()
        content_len = next((k for k, tok in enumerate(completion_ids) if tok in special_ids), len(completion_ids))
        completion_texts.append(student_tok.decode(completion_ids[:content_len], clean_up_tokenization_spaces=False))

    teacher_input_ids, teacher_labels, _, teacher_spans = build_teacher_inputs(
        teacher_tok, teacher_tok(prompt_texts)["input_ids"], completion_texts
    )
    return teacher_input_ids, teacher_labels, completion_texts, teacher_spans


def _assert_alignment_covers_completion(loss_fn, batch, teacher_spans):
    """Assert the alignment pairs every student token up to its end-of-turn, and every teacher token."""
    for idx in range(batch["input_ids"].shape[0]):
        student_ids = batch["input_ids"][idx][batch["labels"][idx] != -100].tolist()
        student_groups, teacher_groups = loss_fn._align_tokens(student_ids, teacher_spans[idx])
        end_of_turn = next(k for k, tok in enumerate(student_ids) if tok in loss_fn.student_special_ids)
        assert sorted(k for group in student_groups for k in group) == list(range(end_of_turn + 1))
        assert sorted(k for group in teacher_groups for k in group) == list(range(len(teacher_spans[idx])))


def _check_uld_on_batch(batch, student_tokenizer, teacher_tokenizer, assistant_texts):
    """The teacher's completions hold the assistant text, the alignment covers them, and the hybrid ULD loss on random
    logits is finite."""
    teacher_input_ids, teacher_labels, completion_texts, teacher_spans = _teacher_inputs_from_collator(
        student_tokenizer, teacher_tokenizer, batch
    )
    for completion, assistant in zip(completion_texts, assistant_texts, strict=True):
        assert assistant.strip() in completion

    config = build_config(uld_use_hybrid_loss=True, uld_hybrid_matched_weight=0.5, uld_hybrid_unmatched_weight=0.5)
    loss_fn = ULDLoss(config, student_tokenizer=student_tokenizer, teacher_tokenizer=teacher_tokenizer)
    _assert_alignment_covers_completion(loss_fn, batch, teacher_spans)

    torch.manual_seed(0)
    batch_size, seq_len = batch["input_ids"].shape
    loss = loss_fn(
        student_logits=torch.randn(batch_size, seq_len, len(student_tokenizer)),
        teacher_logits=torch.randn(batch_size, teacher_input_ids.shape[1], len(teacher_tokenizer)),
        student_labels=batch["labels"],
        teacher_labels=teacher_labels,
        student_input_ids=batch["input_ids"],
        teacher_input_ids=teacher_input_ids,
        teacher_spans=teacher_spans,
    )
    assert torch.isfinite(loss)


@pytest.mark.parametrize(
    "student, examples",
    [
        ("llama_tokenizer", "openr1_examples"),
        ("llama_tokenizer", "countdown_examples"),
        ("smollm_tokenizer", "openr1_examples"),
    ],
)
def test_chatml_collator_preserves_completion(request, qwen_tokenizer, student, examples):
    student_tokenizer = request.getfixturevalue(student)
    examples = request.getfixturevalue(examples)
    batch = DataCollatorForChatML(tokenizer=student_tokenizer, max_length=512)(examples)

    assistant_texts = [example["messages"][-1]["content"] for example in examples]
    decoded_batch = student_tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=False)
    for decoded, assistant in zip(decoded_batch, assistant_texts, strict=True):
        assert assistant.strip() in decoded

    _check_uld_on_batch(batch, student_tokenizer, qwen_tokenizer, assistant_texts)


def build_config(**overrides):
    return GOLDConfig(uld_skip_student_eos=False, uld_skip_teacher_eos=False, **overrides)


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
    """Build input_ids and labels for a (prompt, completion) pair, closing the completion with EOS."""
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    return prompt_ids + completion_ids, [-100] * len(prompt_ids) + completion_ids


def pad_tokens(ids, pad_id, target_length):
    return ids + [pad_id] * (target_length - len(ids))


def pad_labels(labels, target_length):
    return labels + [-100] * (target_length - len(labels))


def _gold_trainer(tmp_path, **config_kwargs):
    return GOLDTrainer(
        model=_TINY_QWEN2,
        teacher_model=_TINY_QWEN2,
        args=GOLDConfig(output_dir=str(tmp_path), report_to="none", **config_kwargs),
        train_dataset=load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train"),
    )


class RecordingVLLMGeneration:
    """Stands in for vLLM, which cannot run here: records the prompts it gets and completes each one with "C"."""

    def __init__(self, processing_class, **kwargs):
        self.completion_ids = processing_class.encode("C", add_special_tokens=False)
        self.prompts = None
        self.sync_calls = 0

    def sync_weights(self):
        self.sync_calls += 1

    def generate(self, prompts, images, num_generations):
        self.prompts = prompts
        return None, [self.completion_ids] * len(prompts), None, None


def _gold_vllm_trainer(monkeypatch, tmp_path, **config_kwargs):
    monkeypatch.setattr(gold_trainer_module, "is_vllm_available", lambda: True)
    monkeypatch.setattr(gold_trainer_module, "VLLMGeneration", RecordingVLLMGeneration)
    return _gold_trainer(tmp_path, use_vllm=True, **config_kwargs)


def test_process_completions_to_buffer_left_pads_prompt_ids(tmp_path):
    trainer = _gold_trainer(tmp_path)
    trainer._buffered_inputs = [None]
    trainer._buffered_text_logs = [None]
    pad = trainer.processing_class.pad_token_id

    trainer._process_completions_to_buffer(
        slices=[{"slice": "original"}],
        on_policy_indices=[0],
        local_slice_indices=[0, 0],
        completion_ids=[[31], [41]],
        prompt_ids_list=[[11], [21, 22]],
        prompts_text=["short", "longer"],
        max_completion_length=1,
    )

    buffered_inputs = trainer._buffered_inputs[0]
    assert buffered_inputs["input_ids"].tolist() == [[pad, 11, 31], [21, 22, 41]]
    assert buffered_inputs["attention_mask"].tolist() == [[0, 1, 1], [1, 1, 1]]
    assert buffered_inputs["labels"].tolist() == [[-100, -100, 31], [-100, -100, 41]]


def test_generate_on_policy_for_slices_uses_prompt_attention_mask_for_vllm_prompts(monkeypatch, tmp_path):
    # The prompt contains the pad token as a real token: the attention mask, not the token id, says what to keep
    trainer = _gold_vllm_trainer(monkeypatch, tmp_path, max_length=None, max_completion_length=1)
    trainer._buffered_inputs = [None]
    trainer._buffered_text_logs = [None]
    tokenizer = trainer.processing_class
    pad = tokenizer.pad_token_id
    a, b = tokenizer.convert_tokens_to_ids(["A", "B"])
    completion_ids = trainer.vllm_generation.completion_ids

    slices = [
        {
            "prompts": torch.tensor([[pad, pad, a, pad, b]]),
            "prompt_attention_mask": torch.tensor([[0, 0, 1, 1, 1]]),
        }
    ]

    trainer._generate_on_policy_for_slices(slices, [0])

    assert trainer.vllm_generation.prompts == [[a, pad, b]]
    assert trainer.vllm_generation.sync_calls == 1
    buffered_inputs = trainer._buffered_inputs[0]
    assert buffered_inputs["input_ids"].tolist() == [[a, pad, b, *completion_ids]]


def test_generate_on_policy_for_slices_reconstructs_prompt_with_special_tokens(monkeypatch, tmp_path):
    trainer = _gold_vllm_trainer(monkeypatch, tmp_path, max_length=None, max_completion_length=1)
    trainer._buffered_inputs = [None]
    trainer._buffered_text_logs = [None]
    tokenizer = trainer.processing_class
    pad = tokenizer.pad_token_id
    a, special, b = tokenizer.convert_tokens_to_ids(["A", "<|im_start|>", "B"])
    completion_ids = trainer.vllm_generation.completion_ids

    slices = [
        {
            "slice": "original",
            "prompts": torch.tensor([[pad, pad, a, special, b]]),
            "prompt_attention_mask": torch.tensor([[0, 0, 1, 1, 1]]),
        }
    ]

    trainer._generate_on_policy_for_slices(slices, [0])

    buffered_inputs = trainer._buffered_inputs[0]
    assert trainer.vllm_generation.prompts == [[a, special, b]]
    assert trainer.vllm_generation.sync_calls == 1
    assert buffered_inputs["input_ids"].tolist() == [[a, special, b, *completion_ids]]
    assert buffered_inputs["attention_mask"].tolist() == [[1, 1, 1, 1]]
    assert buffered_inputs["labels"].tolist() == [[-100, -100, -100, *completion_ids]]
    assert trainer._buffered_text_logs[0] == (["A<|im_start|>B"], ["C"])


def test_on_policy_truncates_prompt_keeping_its_end(monkeypatch, tmp_path):
    """When the prompt overflows max_length - max_completion_length it is truncated before the student sees it."""
    trainer = _gold_vllm_trainer(monkeypatch, tmp_path, max_length=3, max_completion_length=1)
    trainer._buffered_inputs = [None]
    trainer._buffered_text_logs = [None]
    tokenizer = trainer.processing_class
    pad = tokenizer.pad_token_id
    a, special, b = tokenizer.convert_tokens_to_ids(["A", "<|im_start|>", "B"])
    completion_ids = trainer.vllm_generation.completion_ids

    slices = [
        {
            "prompts": torch.tensor([[pad, pad, a, special, b]]),
            "prompt_attention_mask": torch.tensor([[0, 0, 1, 1, 1]]),
        }
    ]

    trainer._generate_on_policy_for_slices(slices, [0])

    # prompt_max_length = max_length - max_new_tokens = 3 - 1 = 2; budgeted BEFORE generation, keeping the
    # END of the prompt (the generation marker), so [A, <|im_start|>, B] -> [<|im_start|>, B] and the student
    # generates from it.
    assert trainer.vllm_generation.prompts == [[special, b]]

    buffered_inputs = trainer._buffered_inputs[0]
    assert buffered_inputs["input_ids"].tolist() == [[special, b, *completion_ids]]
    assert buffered_inputs["prompts"].tolist() == [[special, b]]
    assert buffered_inputs["prompt_attention_mask"].tolist() == [[1, 1]]


def test_non_vllm_on_policy_budgets_prompt_before_generation(tmp_path):
    """Non-vLLM path budgets the prompt to max_length - max_new_tokens (keep-end) before model.generate,
    matching the vLLM path, so the student generates from and trains on the same in-budget context."""
    trainer = _gold_trainer(tmp_path, max_length=3, max_completion_length=1)
    trainer._buffered_inputs = [None]
    trainer._buffered_text_logs = [None]
    tokenizer = trainer.processing_class
    pad = tokenizer.pad_token_id
    a, special, b = tokenizer.convert_tokens_to_ids(["A", "<|im_start|>", "B"])
    device = trainer.accelerator.device

    slices = [
        {
            "prompts": torch.tensor([[pad, a, special, b]], device=device),
            "prompt_attention_mask": torch.tensor([[0, 1, 1, 1]], device=device),
        }
    ]

    trainer._generate_non_vllm_for_slices(slices, [0])

    buffered_inputs = trainer._buffered_inputs[0]
    # prompt_max_length = 3 - 1 = 2; keep-end of the width-4 prompt -> [<|im_start|>, B], then generate one token.
    assert buffered_inputs["prompts"].tolist() == [[special, b]]
    assert buffered_inputs["input_ids"].shape == (1, 3)
    assert buffered_inputs["input_ids"][:, :2].tolist() == [[special, b]]


def test_non_vllm_on_policy_does_not_trim_padded_width_when_real_prompt_fits(tmp_path):
    """Fallback budgets on the REAL token count, not the padded width (vLLM parity). A left-padded prompt
    whose real tokens already fit max_length - max_new_tokens is passed through unchanged, not trimmed to the last
    `budget` columns (which would carry a pad column into generation)."""
    trainer = _gold_trainer(tmp_path, max_length=3, max_completion_length=1)
    trainer._buffered_inputs = [None]
    trainer._buffered_text_logs = [None]
    tokenizer = trainer.processing_class
    pad = tokenizer.pad_token_id
    q = tokenizer.convert_tokens_to_ids("Q")
    device = trainer.accelerator.device

    slices = [
        {
            "prompts": torch.tensor([[pad, pad, q]], device=device),
            "prompt_attention_mask": torch.tensor([[0, 0, 1]], device=device),
        }
    ]

    trainer._generate_non_vllm_for_slices(slices, [0])

    buffered_inputs = trainer._buffered_inputs[0]
    assert buffered_inputs["prompts"].tolist() == [[pad, pad, q]]
    assert buffered_inputs["input_ids"].shape == (1, 4)
    assert buffered_inputs["input_ids"][:, :3].tolist() == [[pad, pad, q]]


def test_gold_trainer_init_defaults_vllm_max_model_length_to_max_length(monkeypatch, tmp_path):
    captured = {}

    # Stand in for vLLM, which cannot run here
    class CapturingVLLMGeneration:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(gold_trainer_module, "is_vllm_available", lambda: True)
    monkeypatch.setattr(gold_trainer_module, "VLLMGeneration", CapturingVLLMGeneration)

    GOLDTrainer(
        model=_TINY_QWEN2,
        teacher_model=_TINY_QWEN2,
        args=GOLDConfig(
            output_dir=str(tmp_path), report_to="none", max_length=128, max_completion_length=16, use_vllm=True
        ),
        train_dataset=load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train"),
    )

    assert captured["max_model_length"] == 128


def test_chatml_collator_truncates_keeping_completion_end(llama_tokenizer):
    """When the rendered chat-template message exceeds max_length, the collator must
    keep the LAST max_length tokens (the model's recent context), not the first."""
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
    formatted_message = llama_tokenizer.apply_chat_template(
        examples[0]["messages"], add_generation_prompt=False, tokenize=False
    )
    full_ids = llama_tokenizer(formatted_message, add_special_tokens=False)["input_ids"]
    assert batch["input_ids"][0, -1].item() == full_ids[-1]


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


def _prepared_row(tmp_path, model_id, tokenizer, dataset, **config_kwargs):
    # `GOLDTrainer.__init__` prepares the train dataset, so its first row is what training sees
    trainer = GOLDTrainer(
        model=model_id,
        teacher_model=model_id,
        args=GOLDConfig(output_dir=str(tmp_path), report_to="none", **config_kwargs),
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    return trainer.train_dataset[0]


def test_prepared_tokenized_rows_keep_completion_after_truncation(tmp_path, llama_tokenizer):
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
    row = _prepared_row(
        tmp_path,
        _TINY_LLAMA,
        llama_tokenizer,
        dataset,
        max_length=max_length,
        max_completion_length=16,
        use_extended_uld=True,
    )

    assert len(row["input_ids"]) == max_length  # truncated, not dropped
    assert 1 in row["completion_mask"]  # completion survived front-truncation

    collator = DataCollatorForChatML(tokenizer=llama_tokenizer, max_length=max_length)
    batch = collator([row])

    # The collator labels exactly the tracked completion tokens — never all -100 — and the assistant text survives.
    completion_ids = [tid for tid, m in zip(row["input_ids"], row["completion_mask"], strict=False) if m == 1]
    supervised = [label for label in batch["labels"][0].tolist() if label != -100]
    assert supervised == completion_ids
    assert assistant in llama_tokenizer.decode(completion_ids)


def _uld_trainer(tmp_path, dataset, model=_TINY_QWEN2, **config_kwargs):
    # A ChatML student (Qwen2 by default) distilled from a teacher with the Llama-3 chat template
    return GOLDTrainer(
        model=model,
        teacher_model=_TINY_LLAMA,
        args=GOLDConfig(
            output_dir=str(tmp_path),
            report_to="none",
            use_uld_loss=True,
            teacher_tokenizer_name_or_path=_TINY_LLAMA,
            **config_kwargs,
        ),
        train_dataset=dataset,
    )


def test_uld_teacher_reads_the_prompt_in_its_own_chat_template(tmp_path, llama_tokenizer):
    messages = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello there"}]
    trainer = _uld_trainer(
        tmp_path, Dataset.from_dict({"messages": [messages]}), max_length=128, max_completion_length=16
    )

    row = trainer.train_dataset[0]

    teacher_prompt = llama_tokenizer.apply_chat_template(messages[:1], add_generation_prompt=True, tokenize=False)
    assert row["teacher_prompt_ids"] == llama_tokenizer(teacher_prompt, add_special_tokens=False)["input_ids"]
    assert row["teacher_prompt_ids"].count(llama_tokenizer.bos_token_id) == 1
    assert "<|im_start|>" not in llama_tokenizer.decode(row["teacher_prompt_ids"])


@pytest.mark.parametrize(
    "max_length, max_completion_length",
    [
        # The prompt is 45 tokens and the whole row 50
        pytest.param(40, 8, id="row_longer_than_max_length"),
        pytest.param(64, 30, id="prompt_leaves_no_room_to_generate"),
    ],
)
def test_uld_rejects_rows_that_do_not_fit(tmp_path, max_length, max_completion_length):
    """Truncating a row would leave the student less context than the teacher, which reads the whole prompt in its
    own chat template, so cross-tokenizer ULD rejects rows that don't fit instead."""
    messages = [
        {"role": "user", "content": "Please summarize: " + "very long context. " * 3},
        {"role": "assistant", "content": "A summary."},
    ]
    with pytest.raises(ValueError, match="does not fit"):
        _uld_trainer(
            tmp_path,
            Dataset.from_dict({"messages": [messages]}),
            max_length=max_length,
            max_completion_length=max_completion_length,
        )


def test_uld_teacher_does_not_see_the_student_template_inside_the_assistant_turn(tmp_path, llama_tokenizer):
    """Qwen3's template opens every assistant turn with an empty think block. That is the student's template, not the
    completion, so the teacher must read only the content after it, then close the turn with its own EOS."""
    messages = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello there"}]
    trainer = _uld_trainer(
        tmp_path,
        Dataset.from_dict({"messages": [messages]}),
        model="trl-internal-testing/tiny-Qwen3ForCausalLM",
        max_length=128,
        max_completion_length=16,
    )
    teacher_inputs = []
    trainer.teacher_model.register_forward_hook(
        lambda module, args, kwargs, output: teacher_inputs.append(kwargs["input_ids"]), with_kwargs=True
    )

    batch = trainer.data_collator([trainer.train_dataset[0]])
    device = trainer.accelerator.device
    batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    trainer.compute_loss(trainer.model, batch)

    teacher_text = llama_tokenizer.decode(teacher_inputs[0][0])
    assert "<think>" not in teacher_text
    assert teacher_text.endswith("<|start_header_id|>assistant<|end_header_id|>\n\nHello there<|eot_id|>")


def test_prepare_dataset_messages_uses_last_assistant_turn(tmp_path, qwen_tokenizer):
    messages = [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "First?"},
        {"role": "assistant", "content": "One."},
        {"role": "user", "content": "Second?"},
        {"role": "assistant", "content": "Two."},
    ]
    dataset = Dataset.from_dict({"messages": [messages]})
    row = _prepared_row(tmp_path, _TINY_QWEN2, qwen_tokenizer, dataset, max_length=512, use_extended_uld=True)
    expected_prompt = qwen_tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=False)
    expected_full = qwen_tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)

    prompt_ids = [tid for tid, mask in zip(row["input_ids"], row["completion_mask"], strict=True) if mask == 0]
    assert qwen_tokenizer.decode(prompt_ids, skip_special_tokens=False) == expected_prompt

    completion_ids = [tid for tid, mask in zip(row["input_ids"], row["completion_mask"], strict=True) if mask == 1]
    decoded_completion = qwen_tokenizer.decode(
        completion_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    assert decoded_completion == expected_full[len(expected_prompt) :]
    assert "One." not in decoded_completion
    assert "Two." in decoded_completion


def test_prepare_dataset_keeps_seam_token(tmp_path, qwen_tokenizer):
    dataset = Dataset.from_dict({"prompt": ["Question: "], "completion": ["Answer."]})
    row = _prepared_row(
        tmp_path, _TINY_QWEN2, qwen_tokenizer, dataset, max_length=64, max_completion_length=16, use_extended_uld=True
    )

    completion_ids = [
        token_id for token_id, mask in zip(row["input_ids"], row["completion_mask"], strict=True) if mask == 1
    ]
    assert qwen_tokenizer.decode(completion_ids[:-1]).lstrip() == "Answer."
    assert completion_ids.count(qwen_tokenizer.eos_token_id) == 1

    teacher_input_ids, teacher_labels, _, _ = build_teacher_inputs(
        qwen_tokenizer, qwen_tokenizer(["Question: "])["input_ids"], ["Answer."]
    )
    teacher_completion_ids = teacher_input_ids[0][teacher_labels[0] != -100].tolist()
    assert teacher_completion_ids.count(qwen_tokenizer.eos_token_id) == 1


def test_prepare_dataset_positional_uld_supports_sentencepiece(tmp_path, gemma4_tokenizer, qwen_tokenizer):
    dataset = Dataset.from_dict({"text": ["Question: Answer."], "prompt": ["Question: "], "completion": ["Answer."]})
    row = _prepared_row(
        tmp_path,
        "trl-internal-testing/tiny-Gemma4ForConditionalGeneration",
        gemma4_tokenizer,
        dataset,
        max_length=64,
        max_completion_length=16,
        use_extended_uld=False,
    )

    completion_ids = [
        token_id for token_id, mask in zip(row["input_ids"], row["completion_mask"], strict=True) if mask == 1
    ]
    assert completion_ids[-1] == gemma4_tokenizer.eos_token_id
    assert gemma4_tokenizer.decode(completion_ids[:-1]).lstrip() == "Answer."

    teacher_input_ids, teacher_labels, _, _ = build_teacher_inputs(
        qwen_tokenizer, qwen_tokenizer(["Question: "])["input_ids"], ["Answer."]
    )
    teacher_completion_ids = teacher_input_ids[0][teacher_labels[0] != -100].tolist()
    assert qwen_tokenizer.decode(teacher_completion_ids) == "Answer." + qwen_tokenizer.eos_token


def test_build_teacher_inputs_supports_sentencepiece(gemma4_tokenizer):
    input_ids, labels, _, _ = build_teacher_inputs(
        gemma4_tokenizer, gemma4_tokenizer(["Question: "])["input_ids"], ["Answer."]
    )

    completion_ids = input_ids[0][labels[0] != -100].tolist()
    assert completion_ids.count(gemma4_tokenizer.eos_token_id) == 1
    assert gemma4_tokenizer.decode(completion_ids) == "Answer." + gemma4_tokenizer.eos_token


class _NoBackendTokenizer:
    """Wraps a tokenizer but hides `backend_tokenizer`, standing in for a slow tokenizer with no fast backend."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def __getattr__(self, name):
        if name == "backend_tokenizer":
            raise AttributeError(name)
        return getattr(self._tokenizer, name)

    def __call__(self, *args, **kwargs):
        return self._tokenizer(*args, **kwargs)


def test_build_teacher_inputs_works_without_backend_tokenizer(gemma4_tokenizer):
    slow_tokenizer = _NoBackendTokenizer(gemma4_tokenizer)

    input_ids, labels, _, _ = build_teacher_inputs(
        slow_tokenizer, slow_tokenizer(["Question: "])["input_ids"], ["Answer."]
    )

    completion_ids = input_ids[0][labels[0] != -100].tolist()
    assert gemma4_tokenizer.decode(completion_ids) == "Answer." + gemma4_tokenizer.eos_token


def test_prepare_dataset_positional_uld_works_without_backend_tokenizer(tmp_path, gemma4_tokenizer):
    dataset = Dataset.from_dict({"prompt": ["Question: "], "completion": ["Answer."]})
    trainer = GOLDTrainer(
        model="trl-internal-testing/tiny-Gemma4ForConditionalGeneration",
        teacher_model="trl-internal-testing/tiny-Gemma4ForConditionalGeneration",
        args=GOLDConfig(
            output_dir=str(tmp_path),
            report_to="none",
            max_length=64,
            max_completion_length=16,
            use_extended_uld=False,
        ),
        train_dataset=dataset,
        processing_class=gemma4_tokenizer,
    )

    # The trainer only accepts real tokenizer classes, so the wrapper goes straight to the dataset preparation
    row = trainer._prepare_dataset_with_original_text(
        dataset,
        _NoBackendTokenizer(gemma4_tokenizer),
        trainer.args,
        packing=False,
        formatting_func=None,
        dataset_name="train",
    )[0]

    completion_ids = [
        token_id for token_id, mask in zip(row["input_ids"], row["completion_mask"], strict=True) if mask == 1
    ]
    assert gemma4_tokenizer.decode(completion_ids[:-1]).lstrip() == "Answer."


def _paired_texts(loss, student_tokenizer, teacher_tokenizer, student_ids):
    """Align a student completion with the teacher's tokenization of its text, as `GOLDTrainer.compute_loss` does: the
    teacher reads the completion up to its first special token, then appends its own EOS. Returns the decoded text of
    each aligned (student, teacher) group."""
    content_len = next((k for k, tok in enumerate(student_ids) if tok in loss.student_special_ids), len(student_ids))
    completion_text = student_tokenizer.decode(student_ids[:content_len], clean_up_tokenization_spaces=False)
    teacher_input_ids, teacher_labels, _, teacher_spans = build_teacher_inputs(
        teacher_tokenizer, [[]], [completion_text]
    )
    teacher_ids = teacher_input_ids[0][teacher_labels[0] != -100].tolist()

    student_groups, teacher_groups = loss._align_tokens(student_ids, teacher_spans[0])

    decode = partial(type(student_tokenizer).decode, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return [
        (
            decode(student_tokenizer, [student_ids[k] for k in s]),
            decode(teacher_tokenizer, [teacher_ids[k] for k in t]),
        )
        for s, t in zip(student_groups, teacher_groups, strict=True)
    ]


@pytest.mark.parametrize(
    "text", ["SmolLM3-3B says hi \U0001f60a to \u4f60\u597d.", "  leading spaces\n\nand\ttabs", "na\u00efve caf\u00e9"]
)
def test_alignment_pairs_every_token_of_clean_text(llama_tokenizer, qwen_tokenizer, text):
    loss = ULDLoss(build_config(), student_tokenizer=llama_tokenizer, teacher_tokenizer=qwen_tokenizer)
    student_ids = llama_tokenizer(text, add_special_tokens=False)["input_ids"] + [llama_tokenizer.eos_token_id]

    paired = _paired_texts(loss, llama_tokenizer, qwen_tokenizer, student_ids)

    assert "".join(student_text for student_text, _ in paired) == text + llama_tokenizer.eos_token
    # The two end-of-turn tokens pair with each other, not with text.
    assert paired[-1] == (llama_tokenizer.eos_token, qwen_tokenizer.eos_token)


def test_alignment_recovers_after_an_invalid_byte(llama_tokenizer, qwen_tokenizer):
    """A sampled completion can hold a stray byte token. The teacher only sees the decoded text, where it became a
    U+FFFD, so the alignment has to resync right after it instead of drifting for the rest of the completion."""
    loss = ULDLoss(build_config(), student_tokenizer=llama_tokenizer, teacher_tokenizer=qwen_tokenizer)
    stray_byte = llama_tokenizer.convert_tokens_to_ids("\u00c3")
    student_ids = (
        llama_tokenizer("hi", add_special_tokens=False)["input_ids"]
        + [stray_byte]
        + llama_tokenizer(" world and more text", add_special_tokens=False)["input_ids"]
    )

    paired = _paired_texts(loss, llama_tokenizer, qwen_tokenizer, student_ids)

    assert all(student_text == teacher_text for student_text, teacher_text in paired)
    assert "".join(student_text for student_text, _ in paired) == "hi\ufffd world and more text"


def test_alignment_handles_a_completion_cut_mid_character(llama_tokenizer, qwen_tokenizer):
    loss = ULDLoss(build_config(), student_tokenizer=llama_tokenizer, teacher_tokenizer=qwen_tokenizer)
    # The first byte of a 3-byte CJK character: generation stopped before the character was complete.
    student_ids = llama_tokenizer("hello", add_special_tokens=False)["input_ids"] + [
        llama_tokenizer.convert_tokens_to_ids("\u00e4")
    ]

    paired = _paired_texts(loss, llama_tokenizer, qwen_tokenizer, student_ids)

    assert paired[-1] == ("\ufffd", "\ufffd")


def test_alignment_survives_a_normalizing_teacher_tokenizer(llama_tokenizer, qwen_tokenizer):
    """Qwen's tokenizer NFC-normalizes its input, so the teacher's tokens spell a precomposed "e" with acute accent
    where the student's text has "e" plus a combining accent. That one character must not throw off the alignment for
    the rest of the completion."""
    assert qwen_tokenizer.backend_tokenizer.normalizer is not None
    loss = ULDLoss(build_config(), student_tokenizer=llama_tokenizer, teacher_tokenizer=qwen_tokenizer)
    student_ids = llama_tokenizer("cafe\u0301 is open today and tomorrow", add_special_tokens=False)["input_ids"]

    paired = _paired_texts(loss, llama_tokenizer, qwen_tokenizer, student_ids)

    # Only the combining accent, which the teacher's offsets leave out, stays unpaired.
    assert paired == [
        ("ca", "ca"),
        ("fe", "f\u00e9"),
        (" is", " is"),
        (" open", " open"),
        (" today", " today"),
        (" and", " and"),
        (" tomorrow", " tomorrow"),
    ]


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


def test_compute_distillation_loss_bayesian_skips_zero_start_span_extended(llama_tokenizer, qwen_tokenizer):
    # Same zero-start scenario on the extended path: the first aligned group pair is dropped instead of wrapping
    # around, so the loss is finite.
    config = build_config(use_extended_uld=True, uld_token_merge_strategy="bayesian")
    loss_fn = ULDLoss(config, student_tokenizer=llama_tokenizer, teacher_tokenizer=qwen_tokenizer)

    student_ids = llama_tokenizer("Hello world again", add_special_tokens=False)["input_ids"]
    teacher_encoding = qwen_tokenizer("Hello world again", add_special_tokens=False, return_offsets_mapping=True)
    teacher_ids = teacher_encoding["input_ids"]
    torch.manual_seed(0)
    result = loss_fn._compute_distillation_loss(
        torch.randn(1, len(student_ids), len(llama_tokenizer)),
        torch.randn(1, len(teacher_ids), len(qwen_tokenizer)),
        torch.tensor([student_ids]),  # no prompt: the answer span starts at 0
        torch.tensor([teacher_ids]),
        torch.tensor([student_ids]),
        torch.tensor([teacher_ids]),
        teacher_spans=[teacher_encoding["offset_mapping"]],
    )

    assert torch.isfinite(result)


def test_uldloss_positional_mode_does_not_require_tokenizers():
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
    loss_fn = ULDLoss(build_config(), student_tokenizer=None, teacher_tokenizer=None)

    answers = torch.tensor(
        [
            [-100, -100, -100, 10, 20, 30, -100, -100],
            [-100, 5, 6, 7, -100, -100, -100, -100],
            [-100, -100, -100, -100, -100, -100, -100, -100],
        ]
    )

    starts, sizes = loss_fn._get_start_and_size_answers(answers)

    assert starts == [3, 1, 0]
    assert sizes == [3, 3, 0]


def _llama_trainer(tmp_path, tokenizer):
    return GOLDTrainer(
        model=_TINY_LLAMA,
        teacher_model=_TINY_LLAMA,
        args=GOLDConfig(output_dir=str(tmp_path), report_to="none"),
        train_dataset=load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train"),
        processing_class=tokenizer,
    )


def test_generate_on_policy_outputs_masks_prompt(tmp_path, llama_tokenizer):
    trainer = _llama_trainer(tmp_path, llama_tokenizer)

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

    generation_config = GenerationConfig()
    new_ids, new_mask, new_labels, prompt_texts, completion_texts = trainer.generate_on_policy_outputs(
        DummyModel(),
        {"prompts": prompt_tensor, "prompt_attention_mask": prompt_mask},
        generation_config,
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


def test_generate_on_policy_outputs_pad_equals_eos_keeps_eos(tmp_path, llama_tokenizer):
    # pad == eos here (the fixture ties them): identity-based pad masking used to erase the terminating EOS label on
    # every row and zero attention on pad-id tokens inside the prompt.
    trainer = _llama_trainer(tmp_path, llama_tokenizer)
    eos_id = llama_tokenizer.eos_token_id
    pad_id = llama_tokenizer.pad_token_id
    assert pad_id == eos_id

    # Row 0 stops on EOS and is right-padded; row 1 runs to max_new_tokens (no padding).
    prompts = torch.tensor([[pad_id, 11, eos_id, 13], [pad_id, 11, eos_id, 13]])
    prompt_mask = torch.tensor([[0, 1, 1, 1], [0, 1, 1, 1]])
    completions = torch.tensor([[21, 22, eos_id, pad_id, pad_id], [21, 22, 23, 24, 25]])
    generated_sequence = torch.cat([prompts, completions], dim=1)

    class DummyModel:
        def generate(self, input_ids, attention_mask, generation_config, return_dict_in_generate):
            return SimpleNamespace(sequences=generated_sequence)

    generation_config = GenerationConfig(eos_token_id=eos_id)
    inputs = {"prompts": prompts, "prompt_attention_mask": prompt_mask}
    _, new_attention_mask, new_labels, _, _ = trainer.generate_on_policy_outputs(
        DummyModel(), inputs, generation_config
    )

    prompt_width = prompts.shape[1]
    eos_pos = prompt_width + 2
    # Pad-id tokens inside the prompt keep attention, only real prompt padding is masked
    assert torch.equal(new_attention_mask[:, :prompt_width], prompt_mask)
    # The terminating EOS stays attended and supervised, only the padding after it is masked
    assert new_attention_mask[0, eos_pos] == 1
    assert new_labels[0, eos_pos] == eos_id
    assert torch.all(new_attention_mask[0, eos_pos + 1 :] == 0)
    assert torch.all(new_labels[0, eos_pos + 1 :] == -100)
    # A row that hits max_new_tokens has no padding and stays fully supervised
    assert torch.all(new_attention_mask[1, prompt_width:] == 1)
    assert torch.equal(new_labels[1, prompt_width:], completions[1])
    # Prompt positions never contribute to the loss
    assert torch.all(new_labels[:, :prompt_width] == -100)


def test_generate_on_policy_outputs_without_eos_id_keeps_full_completion(tmp_path, llama_tokenizer):
    # GenerationConfig defaults eos_token_id to None; torch.tensor(None) would raise, so the
    # None guard must keep the whole completion instead of trying to find a stop token.
    trainer = _llama_trainer(tmp_path, llama_tokenizer)

    prompts = torch.tensor([[11, 12, 13], [14, 15, 16]])
    prompt_mask = torch.ones_like(prompts)
    completions = torch.tensor([[21, 22, 23, 24], [25, 26, 27, 28]])
    generated_sequence = torch.cat([prompts, completions], dim=1)

    class DummyModel:
        def generate(self, input_ids, attention_mask, generation_config, return_dict_in_generate):
            return SimpleNamespace(sequences=generated_sequence)

    generation_config = GenerationConfig()
    inputs = {"prompts": prompts, "prompt_attention_mask": prompt_mask}
    _, new_attention_mask, new_labels, _, _ = trainer.generate_on_policy_outputs(
        DummyModel(), inputs, generation_config
    )

    prompt_width = prompts.shape[1]
    # With no stop token nothing is masked in the completion region
    assert torch.all(new_attention_mask[:, prompt_width:] == 1)
    assert torch.all(new_labels[:, prompt_width:] != -100)


def test_build_teacher_inputs_pad_equals_eos_keeps_terminating_eos(llama_tokenizer):
    # A teacher tokenizer with no pad token ties pad to eos; the old pad-id label strip and attention
    # loop then dropped the appended terminating EOS from both supervision and attention.
    tokenizer = copy.deepcopy(llama_tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.pad_token_id == tokenizer.eos_token_id

    # Different-length completions so one row is right-padded past its terminating EOS.
    teacher_input_ids, teacher_labels, teacher_attention_mask, _ = build_teacher_inputs(
        tokenizer, tokenizer(["Hi?", "Explain gravity in detail please"])["input_ids"], ["Hello there!", "OK"]
    )
    eos_id = tokenizer.eos_token_id
    for row in range(teacher_input_ids.size(0)):
        supervised = (teacher_labels[row] != -100).nonzero(as_tuple=True)[0]
        last = supervised[-1]
        # The appended terminating EOS is the last supervised token and stays attended
        assert teacher_input_ids[row, last] == eos_id
        assert teacher_labels[row, last] == eos_id
        assert teacher_attention_mask[row, last]


def test_generate_on_policy_outputs_masks_prompt_smollm(tmp_path, smollm_tokenizer, openr1_examples):
    # There is no tiny SmolLM3 model; tiny Llama 3.2 shares SmolLM3's 128k vocabulary, so it pairs with its tokenizer
    trainer = _llama_trainer(tmp_path, smollm_tokenizer)

    collator = DataCollatorForChatML(tokenizer=smollm_tokenizer)
    batch = collator([openr1_examples[0]])
    batch = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    class DummyModel:
        def generate(self, input_ids, attention_mask, generation_config, return_dict_in_generate):
            assert torch.equal(input_ids, batch["prompts"])
            assert torch.equal(attention_mask, batch["prompt_attention_mask"])
            return SimpleNamespace(sequences=batch["input_ids"])

    generation_config = GenerationConfig()
    new_ids, new_mask, new_labels, prompt_texts, completion_texts = trainer.generate_on_policy_outputs(
        DummyModel(),
        {
            "prompts": batch["prompts"],
            "prompt_attention_mask": batch["prompt_attention_mask"],
        },
        generation_config,
    )

    assert torch.equal(new_ids, batch["input_ids"])
    # With eos_token_id=None the whole completion is kept, so only real prompt padding is masked
    prompt_len_cols = batch["prompts"].shape[1]
    expected_mask = torch.ones_like(batch["input_ids"])
    expected_mask[:, :prompt_len_cols] = batch["prompt_attention_mask"]
    assert torch.equal(new_mask, expected_mask)

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


def _uld_loss_on_text(loss_fn, student_tokenizer, teacher_tokenizer, prompt, completion):
    """ULD loss on random logits for one (prompt, completion), tokenized by each side as the trainer does."""
    student_ids, student_labels = encode_prompt_completion(student_tokenizer, prompt, completion)
    teacher_input_ids, teacher_labels, _, teacher_spans = build_teacher_inputs(
        teacher_tokenizer, teacher_tokenizer([prompt])["input_ids"], [completion]
    )
    max_length = max(len(student_ids), teacher_input_ids.shape[1])
    teacher_input_ids = pad_tokens(teacher_input_ids[0].tolist(), teacher_tokenizer.pad_token_id, max_length)
    teacher_labels = pad_labels(teacher_labels[0].tolist(), max_length)
    return loss_fn(
        student_logits=torch.randn(1, max_length, len(student_tokenizer)),
        teacher_logits=torch.randn(1, max_length, len(teacher_tokenizer)),
        student_labels=torch.tensor([pad_labels(student_labels, max_length)]),
        teacher_labels=torch.tensor([teacher_labels]),
        student_input_ids=torch.tensor([pad_tokens(student_ids, student_tokenizer.pad_token_id, max_length)]),
        teacher_input_ids=torch.tensor([teacher_input_ids]),
        teacher_spans=teacher_spans,
    )


@pytest.mark.parametrize("student", ["llama_tokenizer", "smollm_tokenizer"])
def test_uldloss_handles_cross_tokenizer_sequence(request, qwen_tokenizer, student):
    student_tokenizer = request.getfixturevalue(student)
    config = build_config(uld_use_hybrid_loss=True, uld_hybrid_matched_weight=0.6, uld_hybrid_unmatched_weight=0.4)
    loss_fn = ULDLoss(config, student_tokenizer=student_tokenizer, teacher_tokenizer=qwen_tokenizer)

    loss = _uld_loss_on_text(
        loss_fn,
        student_tokenizer,
        qwen_tokenizer,
        "User: Summarize the difference between llamas and alpacas.",
        "Assistant: Llamas are taller while alpacas have softer wool.",
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
        beta=0.0,
    )
    loss_fn = ULDLoss(config, student_tokenizer=llama_tokenizer, teacher_tokenizer=qwen_tokenizer)

    torch.manual_seed(0)
    loss = _uld_loss_on_text(
        loss_fn,
        llama_tokenizer,
        qwen_tokenizer,
        "User: Explain how GOLD handles tokenizer mismatches.",
        "Assistant: GOLD merges aligned subwords and applies hybrid ULD loss.",
    )

    # With no weight on matched tokens, the loss is the unmatched part alone
    torch.testing.assert_close(loss, loss_fn.last_unmatched_loss, atol=1e-6, rtol=1e-5)


def test_build_teacher_vlm_inputs_feeds_images_and_completion(tmp_path, vlm_dataset, vlm_examples):
    """Cross-architecture VLM ULD must render the teacher prompt through the teacher processor (so it actually sees
    the image via pixel_values), followed by the teacher's own tokenization of the completion and its EOS."""
    with pytest.warns(UserWarning, match="Cross-architecture VLM distillation"):
        trainer = GOLDTrainer(
            model=_TINY_SMOLVLM,
            teacher_model=_TINY_QWEN3_VL,
            args=GOLDConfig(output_dir=str(tmp_path), report_to="none", use_uld_loss=True, use_extended_uld=True),
            train_dataset=vlm_dataset,
        )

    images, prompts = trainer._extract_images_and_prompts(vlm_examples)
    completion_texts = _get_assistant_texts(vlm_examples)

    teacher_input_ids, teacher_labels, teacher_attention_mask, _, teacher_forward_kwargs = (
        trainer._build_teacher_vlm_inputs(completion_texts, images, prompts)
    )

    assert teacher_input_ids.shape == teacher_labels.shape == teacher_attention_mask.shape
    # The teacher actually receives the image, not just text.
    assert "pixel_values" in teacher_forward_kwargs

    teacher_tokenizer = trainer._teacher_processor.tokenizer
    for row, completion_text in enumerate(completion_texts):
        completion_positions = teacher_labels[row] != -100
        # The prompt (image placeholders + text) is masked; the completion is supervised.
        assert not completion_positions[0]
        expected_ids = teacher_tokenizer(completion_text, add_special_tokens=False)["input_ids"]
        assert teacher_input_ids[row][completion_positions].tolist() == expected_ids + [teacher_tokenizer.eos_token_id]


def test_gold_trainer_init_rejects_llm_with_vision_dataset(tmp_path, vlm_dataset):
    """GOLDTrainer should raise ValueError when a text-only model receives a vision dataset."""
    with pytest.raises(ValueError, match="vision-related"):
        GOLDTrainer(
            model=_TINY_QWEN2,
            teacher_model=_TINY_QWEN2,
            args=GOLDConfig(output_dir=str(tmp_path), report_to="none"),
            train_dataset=vlm_dataset,
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


@pytest.mark.parametrize(
    "student, teacher", [("smolvlm_processor", "qwen3_vl_processor"), ("qwen3_vl_processor", "smolvlm_processor")]
)
def test_vlm_chatml_collator_preserves_completion(request, vlm_examples, student, teacher):
    student_processor = request.getfixturevalue(student)
    teacher_processor = request.getfixturevalue(teacher)
    # 2048 to not truncate the completion tokens
    batch = DataCollatorForVisionLanguageChatML(processor=student_processor, max_length=2048)(vlm_examples)
    for key in ("input_ids", "labels", "prompts", "prompt_attention_mask", "pixel_values"):
        assert key in batch

    assistant_texts = _get_assistant_texts(vlm_examples)
    decoded_batch = student_processor.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=False)
    for decoded, assistant in zip(decoded_batch, assistant_texts, strict=True):
        assert assistant in decoded

    _check_uld_on_batch(batch, student_processor.tokenizer, teacher_processor.tokenizer, assistant_texts)


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


def test_gold_trainer_init_rejects_non_vlm_teacher(tmp_path, vlm_dataset):
    """GOLDTrainer should raise ValueError when the student is a VLM but the instantiated teacher is not."""
    teacher_model = AutoModelForCausalLM.from_pretrained(_TINY_QWEN2)

    with pytest.raises(ValueError, match="VLM distillation requires both student and teacher"):
        GOLDTrainer(
            model=_TINY_SMOLVLM,
            teacher_model=teacher_model,
            args=GOLDConfig(output_dir=str(tmp_path), report_to="none"),
            train_dataset=vlm_dataset,
        )


def test_gold_trainer_init_rejects_keep_end_truncation_for_vlm(tmp_path, vlm_dataset):
    """GOLDTrainer should raise ValueError when truncation_mode='keep_end' is used with a VLM."""
    with pytest.warns(FutureWarning, match="keep_end.*deprecated"):
        training_args = GOLDConfig(output_dir=str(tmp_path), report_to="none", truncation_mode="keep_end")
    with pytest.raises(ValueError, match="truncation_mode='keep_end' is not supported for vision-language models"):
        GOLDTrainer(
            model=_TINY_SMOLVLM,
            teacher_model=_TINY_SMOLVLM,
            args=training_args,
            train_dataset=vlm_dataset,
        )


def test_gold_trainer_vlm_vllm_init_uses_identity_collator(monkeypatch, tmp_path, vlm_dataset):
    """When a VLM processor is used with lmbda > 0 and use_vllm=True, GOLDTrainer should use the identity collator
    and store a _vlm_collator for on-the-fly collation. vLLM should be initialized with max_model_length from args.
    """
    captured = {}

    # Stand in for vLLM, which cannot run here
    class CapturingVLLMGeneration:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(gold_trainer_module, "is_vllm_available", lambda: True)
    monkeypatch.setattr(gold_trainer_module, "VLLMGeneration", CapturingVLLMGeneration)

    trainer = GOLDTrainer(
        model=_TINY_SMOLVLM,
        teacher_model=_TINY_SMOLVLM,
        args=GOLDConfig(
            output_dir=str(tmp_path), report_to="none", max_length=128, max_completion_length=16, use_vllm=True
        ),
        train_dataset=vlm_dataset,
    )

    assert captured["max_model_length"] == 128
    # VLM-specific: identity collator + _vlm_collator for on-the-fly use
    assert trainer.data_collator is identity
    assert isinstance(trainer._vlm_collator, DataCollatorForVisionLanguageChatML)


def test_cross_architecture_vlm_without_uld_raises_error(tmp_path, vlm_dataset):
    """When student and teacher have different model_type and use_uld_loss=False, GOLDTrainer should raise
    a ValueError telling the user to enable ULD loss."""
    with pytest.warns(UserWarning, match="Cross-architecture VLM distillation"):
        with pytest.raises(ValueError, match="Cross-architecture VLM distillation.*use_uld_loss=True"):
            GOLDTrainer(
                model=_TINY_SMOLVLM,
                teacher_model=_TINY_QWEN3_VL,
                args=GOLDConfig(output_dir=str(tmp_path), report_to="none"),
                train_dataset=vlm_dataset,
            )


def test_cross_architecture_vlm_with_uld_sets_teacher_processor(tmp_path, vlm_dataset):
    """When student and teacher have different model_type and use_uld_loss=True, GOLDTrainer should store
    the teacher's processor and emit a warning about cross-architecture distillation."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trainer = GOLDTrainer(
            model=_TINY_SMOLVLM,
            teacher_model=_TINY_QWEN3_VL,
            args=GOLDConfig(output_dir=str(tmp_path), report_to="none", use_uld_loss=True),
            train_dataset=vlm_dataset,
        )

    # The teacher's own processor and tokenizer are used for cross-architecture
    assert trainer._teacher_processor.tokenizer.name_or_path == _TINY_QWEN3_VL
    assert trainer.teacher_tokenizer is trainer._teacher_processor.tokenizer
    assert trainer._is_cross_architecture_vlm is True

    # A cross-architecture warning should have been emitted
    cross_arch_warnings = [w for w in caught if "Cross-architecture VLM distillation" in str(w.message)]
    assert len(cross_arch_warnings) == 1
    assert "smolvlm" in str(cross_arch_warnings[0].message)
    assert "qwen3_vl" in str(cross_arch_warnings[0].message)

    # Identity collator and VLM collator should still be set
    assert trainer.data_collator is identity
    assert trainer._vlm_collator is not None


def test_same_architecture_vlm_no_teacher_processor(tmp_path, vlm_dataset):
    """When student and teacher have the same model_type, GOLDTrainer should not load a teacher processor."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trainer = GOLDTrainer(
            model=_TINY_SMOLVLM,
            teacher_model=_TINY_SMOLVLM,
            args=GOLDConfig(output_dir=str(tmp_path), report_to="none"),
            train_dataset=vlm_dataset,
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


def test_same_architecture_vlm_with_uld_sets_teacher_processor(tmp_path, vlm_dataset):
    """ULD VLM distillation should use a teacher processor even when the VLM model_type matches."""
    trainer = GOLDTrainer(
        model=_TINY_SMOLVLM,
        teacher_model=_TINY_SMOLVLM,
        args=GOLDConfig(output_dir=str(tmp_path), report_to="none", use_uld_loss=True),
        train_dataset=vlm_dataset,
    )

    assert trainer._teacher_processor.tokenizer.name_or_path == _TINY_SMOLVLM
    assert trainer._is_cross_architecture_vlm is False
    assert trainer.teacher_tokenizer is trainer._teacher_processor.tokenizer
    assert trainer.data_collator is identity
    assert trainer._vlm_collator is not None


def _gold_vlm_trainer(tmp_path, vlm_dataset, **config_kwargs):
    return GOLDTrainer(
        model=_TINY_SMOLVLM,
        teacher_model=_TINY_SMOLVLM,
        args=GOLDConfig(output_dir=str(tmp_path), report_to="none", **config_kwargs),
        train_dataset=vlm_dataset,
    )


def test_same_architecture_vlm_uld_preserves_raw_images_for_teacher_processor(tmp_path, vlm_dataset):
    trainer = _gold_vlm_trainer(tmp_path, vlm_dataset, use_uld_loss=True, lmbda=0.0, gradient_accumulation_steps=2)
    trainer.model.train()
    generation_batch = [dict(vlm_dataset[0]), dict(vlm_dataset[1])]
    expected_prompt = prepare_multimodal_messages(
        copy.deepcopy(generation_batch[0]["prompt"]), images=[generation_batch[0]["image"]]
    )

    first_slice = trainer._prepare_inputs(generation_batch)

    assert first_slice["_raw_images"] == [[generation_batch[0]["image"]]]
    assert first_slice["_raw_prompts"] == [expected_prompt]
    assert "_gold_vlm_raw_images" in trainer._buffered_inputs[1]
    assert "_gold_vlm_raw_prompts" in trainer._buffered_inputs[1]


def test_on_policy_vlm_vllm_does_not_duplicate_repeated_sampler_batch(monkeypatch, tmp_path, vlm_dataset):
    """The VLM vLLM path must rely on RepeatSampler for `num_generations` duplication.

    `VLLMGeneration.generate` expects the incoming prompt batch to already contain the repeated prompt entries,
    matching the text-only path. Duplicating here again would produce `num_generations ** 2` completions.
    """
    num_generations = 3
    num_slices = 2
    received = {}

    # Stand in for vLLM, which cannot run here
    class RecordingVLLMGeneration:
        def __init__(self, processing_class, **kwargs):
            self.eos_token_id = processing_class.tokenizer.eos_token_id

        def sync_weights(self):
            pass

        def generate(self, prompts, images, num_generations):
            received["prompts"] = prompts
            received["images"] = images
            return None, [[100 + i, self.eos_token_id] for i in range(len(prompts))], None, None

    monkeypatch.setattr(gold_trainer_module, "is_vllm_available", lambda: True)
    monkeypatch.setattr(gold_trainer_module, "VLLMGeneration", RecordingVLLMGeneration)
    trainer = GOLDTrainer(
        model=_TINY_SMOLVLM,
        teacher_model=_TINY_SMOLVLM,
        args=GOLDConfig(
            output_dir=str(tmp_path),
            report_to="none",
            use_vllm=True,
            num_generations=num_generations,
            lmbda=1.0,
            per_device_train_batch_size=num_generations,
            gradient_accumulation_steps=num_slices,
            max_length=None,
            max_completion_length=4,
        ),
        train_dataset=vlm_dataset,
    )
    trainer._buffered_inputs = {}
    trainer._buffered_text_logs = {}

    unique_examples = [dict(vlm_dataset[i]) for i in range(num_slices)]
    sampler = RepeatSampler(
        unique_examples, mini_repeat_count=num_generations, batch_size=len(unique_examples), shuffle=False
    )
    sampled_examples = [unique_examples[i] for i in sampler]
    raw_slices = [sampled_examples[i : i + num_generations] for i in range(0, len(sampled_examples), num_generations)]
    on_policy_indices = list(range(num_slices))

    trainer._generate_on_policy_vlm_raw(raw_slices, on_policy_indices)

    total_sampled_prompts = num_slices * num_generations
    assert len(received["prompts"]) == total_sampled_prompts
    assert len(received["images"]) == total_sampled_prompts
    # Each prompt arrives once per generation, already repeated by the sampler, not duplicated again
    for i, prompt in enumerate(received["prompts"]):
        assert prompt == received["prompts"][i - i % num_generations]

    # Buffers populated for every on-policy slice; the synthetic examples are collated only when consumed
    for slice_idx in on_policy_indices:
        assert "_gold_vlm_lazy_examples" in trainer._buffered_inputs[slice_idx]
        _, completion_texts = trainer._buffered_text_logs[slice_idx]
        assert len(completion_texts) == num_generations

    first_slice = trainer._materialize_vlm_slice(trainer._buffered_inputs[0])
    assert first_slice["input_ids"].shape[0] == num_generations


def test_vlm_uld_custom_collator_missing_raw_fields_raises_clear_error(tmp_path, vlm_dataset):
    trainer = _gold_vlm_trainer(tmp_path, vlm_dataset, use_uld_loss=True)
    device = trainer.accelerator.device

    inputs = {
        "input_ids": torch.ones(1, 2, dtype=torch.long, device=device),
        "attention_mask": torch.ones(1, 2, dtype=torch.long, device=device),
        "teacher_prompt_ids": [[1]],
    }

    with pytest.raises(ValueError, match="requires `_raw_images` and `_raw_prompts`"):
        trainer.compute_loss(trainer.model, inputs)


def test_off_policy_vlm_collates_only_consumed_slice(tmp_path, vlm_dataset):
    trainer = _gold_vlm_trainer(tmp_path, vlm_dataset, lmbda=0.0, gradient_accumulation_steps=2)
    trainer.model.train()
    generation_batch = [dict(vlm_dataset[i]) for i in [0, 1, 2, 0]]

    first_slice = trainer._prepare_inputs(generation_batch)

    assert first_slice["input_ids"].shape[0] == 2
    # The second slice stays raw until it is consumed
    assert "_gold_vlm_lazy_examples" in trainer._buffered_inputs[1]

    second_slice = trainer._prepare_inputs(generation_batch)

    assert second_slice["input_ids"].shape[0] == 2
    assert "_gold_vlm_lazy_examples" not in trainer._buffered_inputs[1]


def test_eval_vlm_collates_raw_batch_off_policy(tmp_path, vlm_dataset):
    """VLM eval (identity collator yields raw dicts) must collate off-policy in `_prepare_inputs`.

    Regression test for the eval crash: the inherited path indexed the raw `list[dict]`. Eval must run the VLM collator
    over the whole batch (no slicing, no buffering, no generation).
    """
    trainer = _gold_vlm_trainer(tmp_path, vlm_dataset)
    trainer.model.eval()
    generation_batch = [dict(vlm_dataset[i]) for i in range(3)]

    inputs = trainer._prepare_inputs(generation_batch)

    # The whole eval batch is collated once (no per-accumulation-step slicing) into a tensor dict.
    assert inputs["input_ids"].shape[0] == len(generation_batch)
    assert "pixel_values" in inputs
    # Off-policy only: no generation occurred, so no on-policy buffer state was created.
    assert trainer._buffered_inputs is None


def test_eval_vlm_attaches_raw_images_for_teacher_processor(tmp_path, vlm_dataset):
    """When a teacher processor is configured (cross-arch / ULD), eval must attach raw images and prompts."""
    trainer = _gold_vlm_trainer(tmp_path, vlm_dataset, use_uld_loss=True)
    trainer.model.eval()
    generation_batch = [dict(vlm_dataset[0]), dict(vlm_dataset[1])]
    expected_prompts = [
        prepare_multimodal_messages(copy.deepcopy(example["prompt"]), images=[example["image"]])
        for example in generation_batch
    ]

    inputs = trainer._prepare_inputs(generation_batch)

    assert inputs["_raw_images"] == [[example["image"]] for example in generation_batch]
    assert inputs["_raw_prompts"] == expected_prompts


def test_on_policy_vlm_without_vllm_collates_only_consumed_slice(tmp_path, vlm_dataset):
    # LFM2-VL's processor returns `spatial_shapes`, which generation needs alongside the pixel values
    trainer = GOLDTrainer(
        model=_TINY_LFM2_VL,
        teacher_model=_TINY_LFM2_VL,
        args=GOLDConfig(
            output_dir=str(tmp_path), report_to="none", gradient_accumulation_steps=2, max_completion_length=2
        ),
        train_dataset=vlm_dataset,
    )
    trainer.model.train()
    # As at the second accumulation step, with both slices still to generate
    trainer._buffered_inputs = [None, None]
    trainer._buffered_text_logs = [None, None]
    trainer._step = 1
    raw_slices = [[dict(vlm_dataset[0])], [dict(vlm_dataset[1])]]

    trainer._generate_on_policy_vlm_raw(raw_slices, [0, 1])

    assert "_gold_vlm_on_policy_raw_examples" in trainer._buffered_inputs[0]
    assert "_gold_vlm_on_policy_raw_examples" in trainer._buffered_inputs[1]

    consumed_slice = trainer._prepare_inputs(raw_slices)

    prompt_length = consumed_slice["prompts"].shape[1]
    completion_ids = consumed_slice["input_ids"][0, prompt_length:]
    assert 1 <= len(completion_ids) <= 2
    assert "spatial_shapes" in consumed_slice
    assert "_gold_vlm_on_policy_raw_examples" in trainer._buffered_inputs[0]
    assert "_gold_vlm_on_policy_raw_examples" not in trainer._buffered_inputs[1]


def test_model_forward_kwargs_preserve_processor_tensor_fields(tmp_path, vlm_dataset):
    trainer = GOLDTrainer(
        model=_TINY_SMOLVLM,
        teacher_model=_TINY_SMOLVLM,
        args=GOLDConfig(output_dir=str(tmp_path), report_to="none"),
        train_dataset=vlm_dataset,
    )
    inputs = {
        "input_ids": torch.ones(1, 2, dtype=torch.long),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
        "labels": torch.ones(1, 2, dtype=torch.long),
        "prompts": torch.ones(1, 1, dtype=torch.long),
        "prompt_attention_mask": torch.ones(1, 1, dtype=torch.long),
        "completion_mask": torch.ones(1, 2, dtype=torch.long),
        "assistant_masks": torch.ones(1, 2, dtype=torch.long),
        "teacher_prompt_ids": [[1]],
        "_raw_images": [vlm_dataset[0]["image"]],
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

_VLM_SMOKE_MAX_LENGTH = 4096


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


def test_generation_suppresses_image_token(tmp_path, vlm_dataset):
    # Gemma3: its processor's `image_token_id` is not the placeholder token, unlike its config's
    trainer = GOLDTrainer(
        model="trl-internal-testing/tiny-Gemma3ForConditionalGeneration",
        teacher_model="trl-internal-testing/tiny-Gemma3ForConditionalGeneration",
        args=GOLDConfig(output_dir=str(tmp_path), report_to="none", use_uld_loss=False),
        train_dataset=vlm_dataset,
    )

    assert trainer.generation_config.suppress_tokens == [trainer.model.config.image_token_id]


@pytest.mark.xfail(
    condition=Version("5.20.1") <= Version(transformers.__version__) < Version("5.21.0.dev0"),
    reason="SmolVLM forward with images is broken in transformers>=5.20.1,<5.21 (see #7790)",
    strict=True,
)
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

    with pytest.warns(UserWarning, match="Cross-architecture VLM distillation"):
        trainer = GOLDTrainer(
            model=student,
            teacher_model=teacher,
            args=args,
            train_dataset=vlm_dataset,
            processing_class=processor,
        )
    train_output = trainer.train()
    assert torch.isfinite(torch.tensor(train_output.training_loss))


def _uld_revision_trainer(tmp_path, teacher_tokenizer_name_or_path):
    args = GOLDConfig(
        output_dir=str(tmp_path),
        report_to="none",
        use_vllm=False,
        use_uld_loss=True,
        teacher_model_revision=_TINY_QWEN2_REVISION,
        teacher_tokenizer_name_or_path=teacher_tokenizer_name_or_path,
    )
    return GOLDTrainer(
        model=_TINY_QWEN2,
        teacher_model=_TINY_QWEN2,
        args=args,
        train_dataset=load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train[:3]"),
    )


def test_uld_teacher_tokenizer_keeps_the_teacher_revision_within_one_repo(tmp_path):
    trainer = _uld_revision_trainer(tmp_path, _TINY_QWEN2)

    assert len(trainer.teacher_tokenizer.chat_template) == 2558


def test_uld_teacher_tokenizer_drops_the_teacher_revision_across_repos(tmp_path):
    """A teacher commit exists only in the teacher's own repo, so pointing ULD's cross-tokenizer setup at another
    repo must not carry it over: `revision` would 404 there.
    """
    trainer = _uld_revision_trainer(tmp_path, _TINY_LLAMA)

    assert trainer.teacher_tokenizer.name_or_path == _TINY_LLAMA


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

    @pytest.mark.parametrize(
        "eval_dataset_type",
        [
            "dataset",
            "iterable_dataset",
            "dataset_dict",
            "iterable_dataset_dict",
            "dict_of_dataset",
            "dict_of_iterable_dataset",
            "none",
        ],
    )
    def test_init_with_eval_dataset(self, eval_dataset_type):
        train_dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")

        if eval_dataset_type == "none":
            eval_dataset = None
        else:
            streaming = "iterable" in eval_dataset_type
            eval_split = load_dataset(
                "trl-internal-testing/zen", "conversational_prompt_completion", split="test", streaming=streaming
            )
            if eval_dataset_type in ("dataset", "iterable_dataset"):
                eval_dataset = eval_split
            elif eval_dataset_type in ("dataset_dict", "iterable_dataset_dict"):
                dataset_dict_cls = IterableDatasetDict if streaming else DatasetDict
                eval_dataset = dataset_dict_cls({"data1": eval_split, "data2": eval_split})
            else:  # "dict_of_dataset" or "dict_of_iterable_dataset"
                eval_dataset = {"data1": eval_split, "data2": eval_split}

        training_args = GOLDConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = GOLDTrainer(
            model=self.model_id,
            teacher_model=self.model_id,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
        )

        if eval_dataset_type == "none":
            assert trainer.eval_dataset is None
        elif isinstance(trainer.eval_dataset, dict):
            assert set(trainer.eval_dataset.keys()) == {"data1", "data2"}
            # Each split was tokenized independently.
            assert "input_ids" in next(iter(trainer.eval_dataset["data1"]))
            assert "input_ids" in next(iter(trainer.eval_dataset["data2"]))
        else:
            assert "input_ids" in next(iter(trainer.eval_dataset))
