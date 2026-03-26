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

from types import SimpleNamespace

import pytest
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from trl.experimental.gold import gold_trainer as gold_trainer_module
from trl.experimental.gold.gold_trainer import GOLDTrainer, ULDLoss, build_teacher_inputs_from_texts
from trl.experimental.utils import DataCollatorForChatML


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

    teacher_input_ids, teacher_labels, _, _ = build_teacher_inputs_from_texts(
        teacher_tok, prompt_texts, completion_texts
    )
    return teacher_input_ids, teacher_labels, completion_texts


def _assert_alignment_covers_completion(loss_fn, batch, teacher_input_ids, teacher_labels):
    for idx in range(batch["input_ids"].shape[0]):
        student_mask = batch["attention_mask"][idx].bool()
        student_ids = batch["input_ids"][idx][student_mask]
        student_labels = batch["labels"][idx][student_mask]
        student_answer_ids = student_ids[student_labels != -100].tolist()

        teacher_answer_mask = teacher_labels[idx] != -100
        teacher_answer_ids = teacher_input_ids[idx][teacher_answer_mask].tolist()

        student_groups, teacher_groups = loss_fn._build_alignment_groups_from_ids(
            student_answer_ids, teacher_answer_ids
        )

        assert student_groups, "Student alignment groups must not be empty"
        assert teacher_groups, "Teacher alignment groups must not be empty"
        assert sorted(idx for group in student_groups for idx in group) == list(range(len(student_answer_ids)))
        assert sorted(idx for group in teacher_groups for idx in group) == list(range(len(teacher_answer_ids)))


@pytest.mark.slow
def test_chatml_collator_preserves_completion_llama(llama_tokenizer, qwen_tokenizer, openr1_examples):
    collator = DataCollatorForChatML(tokenizer=llama_tokenizer, max_length=512)
    batch = collator(openr1_examples)

    assistant_texts = [example["messages"][-1]["content"] for example in openr1_examples]
    decoded_batch = llama_tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=False)
    for decoded, assistant in zip(decoded_batch, assistant_texts, strict=True):
        assert assistant.strip() in decoded

    teacher_input_ids, teacher_labels, completion_texts = _teacher_inputs_from_collator(
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

    _assert_alignment_covers_completion(loss_fn, batch, teacher_input_ids, teacher_labels)

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

    teacher_input_ids, teacher_labels, completion_texts = _teacher_inputs_from_collator(
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

    _assert_alignment_covers_completion(loss_fn, batch, teacher_input_ids, teacher_labels)

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

    teacher_input_ids, teacher_labels, completion_texts = _teacher_inputs_from_collator(
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

    _assert_alignment_covers_completion(loss_fn, batch, teacher_input_ids, teacher_labels)

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
        uld_use_hybrid_loss=False,
        uld_hybrid_matched_weight=None,
        uld_hybrid_unmatched_weight=None,
        beta=0.5,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture(scope="session")
def llama_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
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


def test_process_completions_to_buffer_left_pads_prompt_ids():
    class RecordingTokenizer:
        pad_token_id = 0
        pad_token = "<pad>"

        def batch_decode(self, sequences, skip_special_tokens=False, clean_up_tokenization_spaces=False):
            del skip_special_tokens, clean_up_tokenization_spaces
            return [" ".join(str(token) for token in sequence) for sequence in sequences]

    trainer = GOLDTrainer.__new__(GOLDTrainer)
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
    trainer.processing_class = RecordingTokenizer()
    trainer.args = SimpleNamespace(max_length=None)
    trainer._buffered_inputs = [None]
    trainer._buffered_text_logs = [None]

    GOLDTrainer._process_completions_to_buffer(
        trainer,
        slices=[{"slice": "original"}],
        on_policy_indices=[0],
        local_slice_indices=[0, 0],
        completion_ids=[[31], [41]],
        prompts_text_with_special=["short", "longer"],
        prompt_ids_list=[[11], [21, 22]],
        prompts_text=["short", "longer"],
        max_completion_length=1,
    )

    buffered_inputs = trainer._buffered_inputs[0]
    assert torch.equal(buffered_inputs["input_ids"], torch.tensor([[0, 11, 31], [21, 22, 41]], dtype=torch.long))
    assert torch.equal(buffered_inputs["attention_mask"], torch.tensor([[0, 1, 1], [1, 1, 1]], dtype=torch.long))
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

        def batch_decode(self, sequences, skip_special_tokens=False, clean_up_tokenization_spaces=False):
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

    captured = {}

    def capture_process_completions(
        slices,
        on_policy_indices,
        local_slice_indices,
        completion_ids,
        prompt_ids_list,
        prompts_text_with_special,
        prompts_text,
        max_completion_length,
    ):
        captured["slices"] = slices
        captured["on_policy_indices"] = on_policy_indices
        captured["local_slice_indices"] = local_slice_indices
        captured["completion_ids"] = completion_ids
        captured["prompt_ids_list"] = prompt_ids_list
        captured["prompts_text"] = prompts_text
        captured["prompts_text_with_special"] = prompts_text_with_special
        captured["max_completion_length"] = max_completion_length

    trainer = GOLDTrainer.__new__(GOLDTrainer)
    trainer.accelerator = SimpleNamespace(is_main_process=True)
    trainer.args = SimpleNamespace(report_to=[])
    trainer.processing_class = RecordingTokenizer()
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
    assert captured["prompts_text"] == ["A B"]
    assert captured["prompts_text_with_special"] == ["A <eos> B"]


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

        def batch_decode(self, sequences, skip_special_tokens=False, clean_up_tokenization_spaces=False):
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

    trainer = GOLDTrainer.__new__(GOLDTrainer)
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"), is_main_process=True)
    trainer.processing_class = RecordingTokenizer()
    trainer.args = SimpleNamespace(max_length=None, report_to=[])
    trainer.use_vllm = True
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
    assert torch.equal(buffered_inputs["attention_mask"], torch.tensor([[1, 1, 1, 1]], dtype=torch.long))
    assert torch.equal(buffered_inputs["labels"], torch.tensor([[-100, -100, -100, 42]], dtype=torch.long))
    assert buffered_inputs["original_prompt_text"] == ["A <special> B"]
    assert buffered_inputs["original_completion_text"] == ["C"]
    assert trainer._buffered_text_logs[0] == (["A B"], ["C"])


def test_gold_trainer_init_defaults_vllm_max_model_length_to_max_length(monkeypatch):
    captured = {}

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
        use_transformers_paged=False,
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
    # probs[0] = P(token | context) at position 0 for all vocab tokens
    # probs[1] = P(token | context) at position 1 for all vocab tokens
    probs = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
    loss = ULDLoss(config, student_tokenizer=None, teacher_tokenizer=None)

    # token_ids[1] = 1 means the actual token at position 1 is token ID 1
    # So we should extract P(token_id=1 | ...) = probs[1, 1] = 0.5
    token_ids = [0, 1]  # Actual generated tokens

    merged = loss._merge_probabilities_with_alignment_groups(probs, [[0, 1]], token_ids=token_ids)

    # Expected: P_merged(y) = P(y | context_0) × P(token_1=1 | context_1)
    # For each vocab token y, multiply marginal prob at pos 0 by scalar conditional prob of actual token at pos 1
    expected = probs[0] * probs[1, 1]  # probs[1, 1] = 0.5
    # Expected unnormalized: [0.6 * 0.5, 0.3 * 0.5, 0.1 * 0.5] = [0.3, 0.15, 0.05]

    torch.testing.assert_close(merged[0], expected)


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
    trainer.use_transformers_paged = False
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
    assert torch.equal(new_labels[0, padded_prompt_len:], torch.tensor(completion_ids, dtype=torch.long))

    assert prompt_texts[0] == llama_tokenizer.decode(prompt_ids, skip_special_tokens=False)
    assert completion_texts[0] == llama_tokenizer.decode(completion_ids, skip_special_tokens=False)


@pytest.mark.slow
def test_generate_on_policy_outputs_masks_prompt_smollm(smollm_tokenizer, openr1_examples):
    trainer = GOLDTrainer.__new__(GOLDTrainer)
    trainer.use_transformers_paged = False
    trainer.processing_class = smollm_tokenizer

    collator = DataCollatorForChatML(tokenizer=smollm_tokenizer)
    batch = collator([openr1_examples[0]])
    batch = {k: v.cpu() for k, v in batch.items()}

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
        {"prompts": batch["prompts"], "prompt_attention_mask": batch["prompt_attention_mask"]},
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


def test_uldloss_hybrid_config_beta_zero(llama_tokenizer, qwen_tokenizer):
    config = build_config(
        uld_use_hybrid_loss=True,
        uld_hybrid_matched_weight=0.0,
        uld_hybrid_unmatched_weight=1.0,
        use_extended_uld=True,
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
    )

    assert torch.isfinite(loss)
    assert loss.dim() == 0
    assert loss_fn.last_matched_loss is not None
    assert loss_fn.last_unmatched_loss is not None

    expected = config.uld_hybrid_unmatched_weight * loss_fn.last_unmatched_loss
    torch.testing.assert_close(loss, expected, atol=1e-6, rtol=1e-5)
