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

from functools import partial
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from trl.experimental.gold import GOLDConfig
from trl.experimental.gold import gold_trainer as gold_trainer_module
from trl.experimental.gold.gold_trainer import GOLDTrainer, ULDLoss, build_teacher_inputs_from_texts
from trl.experimental.utils import (
    DataCollatorForChatML,
    encode_with_byte_offsets,
    pad_byte_offsets,
)

from ..testing_utils import TrlTestCase


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


def test_gold_trainer():
    # --- process_completions_to_buffer: left-pads shorter prompts ---
    class _TokBuffer:
        pad_token_id = 0
        pad_token = "<pad>"

        def batch_decode(self, sequences, skip_special_tokens=False, clean_up_tokenization_spaces=False):
            del skip_special_tokens, clean_up_tokenization_spaces
            return [" ".join(str(token) for token in sequence) for sequence in sequences]

        def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
            del skip_special_tokens, clean_up_tokenization_spaces
            return " ".join(str(token) for token in ids)

    trainer = GOLDTrainer.__new__(GOLDTrainer)
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
    trainer.processing_class = _TokBuffer()
    trainer.args = SimpleNamespace(max_length=None)
    trainer.use_uld_loss = False
    trainer.xtoken_loss_fn = None
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

    buffered = trainer._buffered_inputs[0]
    assert torch.equal(buffered["input_ids"], torch.tensor([[0, 11, 31], [21, 22, 41]], dtype=torch.long))
    assert torch.equal(buffered["attention_mask"], torch.tensor([[0, 1, 1], [1, 1, 1]], dtype=torch.long))
    assert torch.equal(buffered["labels"], torch.tensor([[-100, -100, 31], [-100, -100, 41]]))

    # --- generate_on_policy_for_slices: strips padding from vllm prompts via attention mask ---
    class _GenVLLMMask:
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

    class _TokMask:
        pad_token_id = 9
        pad_token = "<eos>"

        def batch_decode(self, sequences, skip_special_tokens=False, clean_up_tokenization_spaces=False):
            del clean_up_tokenization_spaces
            token_map = {5: "A", 6: "B", 9: "<eos>"}
            decoded = []
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

    def _capture_process_completions(
        slices,
        on_policy_indices,
        local_slice_indices,
        completion_ids,
        prompt_ids_list,
        prompts_text,
        max_completion_length,
    ):
        captured["completion_ids"] = completion_ids
        captured["prompt_ids_list"] = prompt_ids_list
        captured["prompts_text"] = prompts_text

    vllm_mask = _GenVLLMMask()
    trainer = GOLDTrainer.__new__(GOLDTrainer)
    trainer.accelerator = SimpleNamespace(is_main_process=True)
    trainer.args = SimpleNamespace(report_to=[])
    trainer.processing_class = _TokMask()
    trainer.use_vllm = True
    trainer.vllm_generation = vllm_mask
    trainer.vllm_sync_frequency = 1
    trainer._last_vllm_sync_step = -1
    trainer.state = SimpleNamespace(global_step=0)
    trainer.num_generations = 1
    trainer.generation_config = SimpleNamespace(max_new_tokens=1)
    trainer._process_completions_to_buffer = _capture_process_completions

    GOLDTrainer._generate_on_policy_for_slices(
        trainer,
        [
            {
                "prompts": torch.tensor([[9, 9, 5, 9, 6]], dtype=torch.long),
                "prompt_attention_mask": torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.long),
            }
        ],
        [0],
    )

    assert vllm_mask.prompts == [[5, 9, 6]]
    assert vllm_mask.sync_calls == 1
    assert captured["completion_ids"] == [[42]]
    assert captured["prompt_ids_list"] == [[5, 9, 6]]
    assert captured["prompts_text"] == ["A <eos> B"]

    # --- generate_on_policy_for_slices: reconstructs prompt with special tokens and writes buffer ---
    class _GenVLLMSpecial:
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

    class _TokSpecial:
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

        def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
            return self.batch_decode([ids], skip_special_tokens, clean_up_tokenization_spaces)[0]

    vllm_special = _GenVLLMSpecial()
    trainer = GOLDTrainer.__new__(GOLDTrainer)
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"), is_main_process=True)
    trainer.processing_class = _TokSpecial()
    trainer.args = SimpleNamespace(max_length=None, report_to=[])
    trainer.use_vllm = True
    trainer.use_uld_loss = False
    trainer.xtoken_loss_fn = None
    trainer.vllm_generation = vllm_special
    trainer.vllm_sync_frequency = 1
    trainer._last_vllm_sync_step = -1
    trainer.state = SimpleNamespace(global_step=0)
    trainer.num_generations = 1
    trainer.generation_config = SimpleNamespace(max_new_tokens=1)
    trainer._buffered_inputs = [None]
    trainer._buffered_text_logs = [None]

    GOLDTrainer._generate_on_policy_for_slices(
        trainer,
        [
            {
                "slice": "original",
                "prompts": torch.tensor([[0, 0, 5, 13, 6]], dtype=torch.long),
                "prompt_attention_mask": torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.long),
            }
        ],
        [0],
    )

    buffered = trainer._buffered_inputs[0]
    assert vllm_special.prompts == [[5, 13, 6]]
    assert vllm_special.sync_calls == 1
    assert torch.equal(buffered["input_ids"], torch.tensor([[5, 13, 6, 42]], dtype=torch.long))
    assert torch.equal(buffered["attention_mask"], torch.tensor([[1, 1, 1, 1]], dtype=torch.long))
    assert torch.equal(buffered["labels"], torch.tensor([[-100, -100, -100, 42]], dtype=torch.long))
    assert buffered["original_prompt_text"] == ["A <special> B"]
    assert buffered["original_completion_text"] == ["C"]
    assert trainer._buffered_text_logs[0] == (["A <special> B"], ["C"])

    # --- generate_on_policy_for_slices: truncates prompt when it overflows max_length ---
    class _GenVLLMTrunc:
        def __init__(self):
            self.prompts = None

        def sync_weights(self):
            pass

        def generate(self, prompts, images, num_generations):
            self.prompts = prompts
            return None, [[42]], None, None

    class _TokTrunc:
        pad_token_id = 0
        pad_token = "<pad>"

        def __init__(self):
            self.truncation_side = "right"

        def batch_decode(self, sequences, skip_special_tokens=False, clean_up_tokenization_spaces=False):
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
    trainer.processing_class = _TokTrunc()
    trainer.args = SimpleNamespace(max_length=3, report_to=[])
    trainer.use_vllm = True
    trainer.use_uld_loss = False
    trainer.xtoken_loss_fn = None
    trainer.teacher_tokenizer = None
    trainer.uld_loss_fn = None
    trainer.vllm_generation = _GenVLLMTrunc()
    trainer.vllm_sync_frequency = 1
    trainer._last_vllm_sync_step = -1
    trainer.state = SimpleNamespace(global_step=0)
    trainer.num_generations = 1
    trainer.generation_config = SimpleNamespace(max_new_tokens=1)
    trainer._buffered_inputs = [None]
    trainer._buffered_text_logs = [None]

    GOLDTrainer._generate_on_policy_for_slices(
        trainer,
        [
            {
                "prompts": torch.tensor([[0, 0, 5, 13, 6]], dtype=torch.long),
                "prompt_attention_mask": torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.long),
            }
        ],
        [0],
    )

    buffered = trainer._buffered_inputs[0]
    # prompt_max_length = max_length - max_completion_length = 3 - 1 = 2; right-truncation keeps [5, 13]
    assert torch.equal(buffered["input_ids"], torch.tensor([[5, 13, 42]], dtype=torch.long))
    assert buffered["original_prompt_text"] == ["A <special>"]

    # --- ULDLoss positional mode: does not require byte offsets ---
    config = build_config(use_extended_uld=False)
    loss_fn = ULDLoss(config, student_tokenizer=None, teacher_tokenizer=None)

    loss = loss_fn(
        student_logits=torch.randn(1, 4, 5),
        teacher_logits=torch.randn(1, 4, 6),
        student_labels=torch.tensor([[-100, 1, 2, -100]]),
        teacher_labels=torch.tensor([[-100, 3, 4, -100]]),
        student_input_ids=torch.tensor([[0, 1, 2, 0]]),
        teacher_input_ids=torch.tensor([[0, 3, 4, 0]]),
    )

    assert torch.isfinite(loss)

    # --- generalized_jsd_loss: accepts probability inputs ---
    student_probs = torch.tensor([[[0.6, 0.3, 0.1]]])
    teacher_probs = torch.tensor([[[0.5, 0.4, 0.1]]])
    mixture = 0.5 * (student_probs + teacher_probs)
    expected_jsd = 0.5 * (
        torch.sum(student_probs * (torch.log(student_probs) - torch.log(mixture)))
        + torch.sum(teacher_probs * (torch.log(teacher_probs) - torch.log(mixture)))
    )

    jsd_loss = GOLDTrainer.generalized_jsd_loss(
        student_probs, teacher_probs, beta=0.5, reduction="batchmean", logits_are_probs=True
    )
    torch.testing.assert_close(jsd_loss, expected_jsd)


def test_gold_trainer_init(monkeypatch):
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
        xtoken_loss_type="none",
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


def test_truncation(llama_tokenizer, qwen_tokenizer):
    # --- collator truncates keeping the end of the completion ---
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

    max_length = 256
    collator = DataCollatorForChatML(tokenizer=llama_tokenizer, max_length=max_length)
    batch = collator(examples)

    assert batch["input_ids"].shape[1] == max_length

    backend = llama_tokenizer.backend_tokenizer
    formatted_message = llama_tokenizer.apply_chat_template(
        examples[0]["messages"], add_generation_prompt=False, tokenize=False
    )
    [(full_ids, _)] = encode_with_byte_offsets(backend, [formatted_message], add_special_tokens=False)
    assert batch["input_ids"][0, -1].item() == full_ids[-1]
    assert tuple(batch["byte_offsets"][0, -1].tolist())[1] > 0

    # --- dataset prep truncates keeping completion; original_prompt/completion_text reflect kept ids ---
    long_user2 = "Please summarize:\n" + ("very long context. " * 200)
    assistant2 = "the short answer"
    dataset = Dataset.from_dict(
        {"messages": [[{"role": "user", "content": long_user2}, {"role": "assistant", "content": assistant2}]]}
    )

    max_length2 = 64
    args2 = SimpleNamespace(
        dataset_num_proc=None,
        dataset_text_field="text",
        max_length=max_length2,
        packing_strategy="bfd",
        use_liger_kernel=False,
    )
    trainer2 = GOLDTrainer.__new__(GOLDTrainer)
    prepared2 = trainer2._prepare_dataset_with_original_text(
        dataset, llama_tokenizer, args2, packing=False, formatting_func=None, dataset_name="train"
    )
    row2 = prepared2[0]

    assert len(row2["input_ids"]) == max_length2
    assert 1 in row2["completion_mask"]

    completion_start = row2["completion_mask"].index(1)
    decode = partial(llama_tokenizer.decode, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    assert row2["original_prompt_text"] == decode(row2["input_ids"][:completion_start])
    assert row2["original_completion_text"] == decode(row2["input_ids"][completion_start:])

    collator2 = DataCollatorForChatML(tokenizer=llama_tokenizer, max_length=max_length2)
    batch2 = collator2([row2])

    completion_ids2 = [tid for tid, m in zip(row2["input_ids"], row2["completion_mask"], strict=False) if m == 1]
    supervised2 = [label for label in batch2["labels"][0].tolist() if label != -100]
    assert supervised2 == completion_ids2
    assert assistant2 in llama_tokenizer.decode(completion_ids2)

    # --- when truncation eats into the completion, byte offsets are rebased to start at 0 ---
    short_prompt = "Q:"
    long_completion = "word " * 300
    dataset3 = Dataset.from_dict({"prompt": [short_prompt], "completion": [long_completion]})

    max_length3 = 32
    args3 = SimpleNamespace(
        dataset_num_proc=None,
        dataset_text_field="text",
        max_length=max_length3,
        packing_strategy="bfd",
        use_liger_kernel=False,
    )
    trainer3 = GOLDTrainer.__new__(GOLDTrainer)
    prepared3 = trainer3._prepare_dataset_with_original_text(
        dataset3, llama_tokenizer, args3, packing=False, formatting_func=None, dataset_name="train"
    )
    row3 = prepared3[0]

    assert len(row3["input_ids"]) == max_length3
    assert row3["completion_mask"] == [1] * max_length3
    assert tuple(row3["byte_offsets"][0]) == (0, len(b"word "))

    # --- dataset preparation uses only the last assistant turn ---
    messages = [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "First?"},
        {"role": "assistant", "content": "One."},
        {"role": "user", "content": "Second?"},
        {"role": "assistant", "content": "Two."},
    ]
    dataset4 = Dataset.from_dict({"messages": [messages]})
    args4 = SimpleNamespace(
        dataset_num_proc=None,
        dataset_text_field="text",
        max_length=512,
        packing_strategy="bfd",
        use_liger_kernel=False,
    )
    trainer4 = GOLDTrainer.__new__(GOLDTrainer)
    prepared4 = trainer4._prepare_dataset_with_original_text(
        dataset4, qwen_tokenizer, args4, packing=False, formatting_func=None, dataset_name="train"
    )
    row4 = prepared4[0]

    expected_prompt = qwen_tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=False)
    expected_full = qwen_tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)

    assert row4["original_prompt_text"] == expected_prompt
    assert row4["original_completion_text"] == expected_full[len(expected_prompt) :]
    assert "One." not in row4["original_completion_text"]
    assert "Two." in row4["original_completion_text"]

    completion_ids4 = [tid for tid, mask in zip(row4["input_ids"], row4["completion_mask"], strict=True) if mask == 1]
    decoded_completion4 = qwen_tokenizer.decode(
        completion_ids4, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    assert decoded_completion4 == row4["original_completion_text"]


def test_uldloss(llama_tokenizer, qwen_tokenizer, smollm_tokenizer):
    def _run_uldloss(config, student_tok, teacher_tok, prompt, completion):
        student_ids, student_labels, student_offsets = encode_prompt_completion(student_tok, prompt, completion)
        teacher_ids, teacher_labels, teacher_offsets = encode_prompt_completion(teacher_tok, prompt, completion)

        max_length = max(len(student_ids), len(teacher_ids))
        student_ids = pad_tokens(student_ids, student_tok.pad_token_id, max_length)
        teacher_ids = pad_tokens(teacher_ids, teacher_tok.pad_token_id, max_length)
        student_labels = pad_labels(student_labels, max_length)
        teacher_labels = pad_labels(teacher_labels, max_length)
        student_byte_offsets = pad_byte_offsets(student_offsets, max_length, padding_side="right").unsqueeze(0)
        teacher_byte_offsets = pad_byte_offsets(teacher_offsets, max_length, padding_side="right").unsqueeze(0)

        loss_fn = ULDLoss(config, student_tokenizer=student_tok, teacher_tokenizer=teacher_tok)
        loss = loss_fn(
            student_logits=torch.randn(1, max_length, len(student_tok)),
            teacher_logits=torch.randn(1, max_length, len(teacher_tok)),
            student_labels=torch.tensor([student_labels]),
            teacher_labels=torch.tensor([teacher_labels]),
            student_input_ids=torch.tensor([student_ids]),
            teacher_input_ids=torch.tensor([teacher_ids]),
            student_byte_offsets=student_byte_offsets,
            teacher_byte_offsets=teacher_byte_offsets,
        )
        return loss, loss_fn

    # llama student, qwen teacher
    config_llama = build_config(
        uld_use_hybrid_loss=True,
        uld_hybrid_matched_weight=0.6,
        uld_hybrid_unmatched_weight=0.4,
    )
    loss, loss_fn = _run_uldloss(
        config_llama,
        llama_tokenizer,
        qwen_tokenizer,
        "User: Summarize the difference between llamas and alpacas.",
        "Assistant: Llamas are taller while alpacas have softer wool.",
    )
    assert torch.isfinite(loss)
    assert loss.dim() == 0
    assert loss_fn.last_matched_loss is not None
    assert loss_fn.last_unmatched_loss is not None

    # smollm student, qwen teacher
    config_smollm = build_config(
        uld_use_hybrid_loss=True,
        uld_hybrid_matched_weight=0.5,
        uld_hybrid_unmatched_weight=0.5,
    )
    loss, loss_fn = _run_uldloss(
        config_smollm,
        smollm_tokenizer,
        qwen_tokenizer,
        "User: Describe SmolLM3 in a sentence.",
        "Assistant: SmolLM3 is a compact yet capable language model.",
    )
    assert torch.isfinite(loss)
    assert loss.dim() == 0
    assert loss_fn.last_matched_loss is not None
    assert loss_fn.last_unmatched_loss is not None

    # hybrid config with matched_weight=0: loss equals unmatched_weight * unmatched_loss
    config_beta0 = build_config(
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
    torch.manual_seed(0)
    loss, loss_fn = _run_uldloss(
        config_beta0,
        llama_tokenizer,
        qwen_tokenizer,
        "User: Explain how GOLD handles tokenizer mismatches.",
        "Assistant: GOLD merges aligned subwords and applies hybrid ULD loss.",
    )
    assert torch.isfinite(loss)
    assert loss.dim() == 0
    assert loss_fn.last_matched_loss is not None
    assert loss_fn.last_unmatched_loss is not None
    expected = config_beta0.uld_hybrid_unmatched_weight * loss_fn.last_unmatched_loss
    torch.testing.assert_close(loss, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.slow
def test_chatml_collator(llama_tokenizer, qwen_tokenizer, smollm_tokenizer, openr1_examples, countdown_examples):
    def _check_collator_and_uldloss(student_tok, teacher_tok, examples, config, seed):
        collator = DataCollatorForChatML(tokenizer=student_tok, max_length=512)
        batch = collator(examples)

        assistant_texts = [ex["messages"][-1]["content"] for ex in examples]
        decoded_batch = student_tok.batch_decode(batch["input_ids"], skip_special_tokens=False)
        for decoded, assistant in zip(decoded_batch, assistant_texts, strict=True):
            assert assistant.strip() in decoded

        teacher_input_ids, teacher_labels, completion_texts, teacher_byte_offsets = _teacher_inputs_from_collator(
            student_tok, teacher_tok, batch
        )
        for completion, assistant in zip(completion_texts, assistant_texts, strict=True):
            assert assistant.strip() in completion
            assert completion.strip()

        torch.manual_seed(seed)
        batch_size, seq_len = batch["input_ids"].shape
        loss_fn = ULDLoss(config, student_tokenizer=student_tok, teacher_tokenizer=teacher_tok)
        loss = loss_fn(
            student_logits=torch.randn(batch_size, seq_len, len(student_tok)),
            teacher_logits=torch.randn(batch_size, teacher_input_ids.shape[1], len(teacher_tok)),
            student_labels=batch["labels"],
            teacher_labels=teacher_labels,
            student_input_ids=batch["input_ids"],
            teacher_input_ids=teacher_input_ids,
            student_byte_offsets=batch["byte_offsets"],
            teacher_byte_offsets=teacher_byte_offsets,
        )
        assert torch.isfinite(loss)

    _check_collator_and_uldloss(
        llama_tokenizer,
        qwen_tokenizer,
        openr1_examples,
        build_config(uld_use_hybrid_loss=True, uld_hybrid_matched_weight=0.6, uld_hybrid_unmatched_weight=0.4),
        seed=0,
    )
    _check_collator_and_uldloss(
        llama_tokenizer,
        qwen_tokenizer,
        countdown_examples,
        build_config(uld_use_hybrid_loss=True, uld_hybrid_matched_weight=0.6, uld_hybrid_unmatched_weight=0.4),
        seed=2,
    )
    _check_collator_and_uldloss(
        smollm_tokenizer,
        qwen_tokenizer,
        openr1_examples,
        build_config(uld_use_hybrid_loss=True, uld_hybrid_matched_weight=0.5, uld_hybrid_unmatched_weight=0.5),
        seed=1,
    )


@pytest.mark.slow
def test_generate_on_policy_outputs(llama_tokenizer, smollm_tokenizer, openr1_examples):
    # llama: manually constructed prompt/completion pair
    prompt_text = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nHello?<|eot_id|>"
    completion_text = "<|start_header_id|>assistant<|end_header_id|>\nHi there!"

    prompt_ids = llama_tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    completion_ids = llama_tokenizer(completion_text, add_special_tokens=False)["input_ids"]

    pad_id = llama_tokenizer.pad_token_id
    pad_width = 3
    prompt_tensor = torch.full((1, len(prompt_ids) + pad_width), pad_id, dtype=torch.long)
    prompt_tensor[0, pad_width:] = torch.tensor(prompt_ids, dtype=torch.long)
    prompt_mask = (prompt_tensor != pad_id).long()
    generated_sequence = torch.cat([prompt_tensor, torch.tensor(completion_ids).unsqueeze(0)], dim=1)

    class _DummyModelLlama:
        def generate(self, input_ids, attention_mask, generation_config, return_dict_in_generate):
            assert torch.equal(input_ids, prompt_tensor)
            assert torch.equal(attention_mask, prompt_mask)
            return SimpleNamespace(sequences=generated_sequence)

    trainer = GOLDTrainer.__new__(GOLDTrainer)
    trainer.processing_class = llama_tokenizer
    generation_config = SimpleNamespace(max_completion_length=None, temperature=None, top_k=None, top_p=None)

    new_ids, new_mask, new_labels, prompt_texts, completion_texts = GOLDTrainer.generate_on_policy_outputs(
        trainer,
        _DummyModelLlama(),
        {"prompts": prompt_tensor, "prompt_attention_mask": prompt_mask},
        generation_config,
        pad_id,
    )

    assert torch.equal(new_ids, generated_sequence)
    if pad_id is not None:
        assert torch.equal(new_mask, (generated_sequence != pad_id).long())
    else:
        assert torch.all(new_mask == 1)

    padded_prompt_len = prompt_tensor.shape[1]
    assert torch.all(new_labels[0, :padded_prompt_len] == -100)
    assert torch.equal(new_labels[0, padded_prompt_len:], torch.tensor(completion_ids, dtype=torch.long))

    unpadded_prompt_ids = prompt_tensor[0][prompt_mask[0].bool()].tolist()
    assert prompt_texts[0] == llama_tokenizer.decode(
        unpadded_prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    assert completion_texts[0] == llama_tokenizer.decode(
        completion_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )

    # smollm: uses a collated real example to verify prompt masking on actual data
    collator = DataCollatorForChatML(tokenizer=smollm_tokenizer)
    batch = collator([openr1_examples[0]])
    batch = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    class _DummyModelSmolLM:
        def generate(self, input_ids, attention_mask, generation_config, return_dict_in_generate):
            assert torch.equal(input_ids, batch["prompts"])
            assert torch.equal(attention_mask, batch["prompt_attention_mask"])
            return SimpleNamespace(sequences=batch["input_ids"])

    trainer2 = GOLDTrainer.__new__(GOLDTrainer)
    trainer2.processing_class = smollm_tokenizer
    pad_id2 = smollm_tokenizer.pad_token_id
    generation_config2 = SimpleNamespace(max_completion_length=None, temperature=None, top_k=None, top_p=None)

    new_ids2, new_mask2, new_labels2, prompt_texts2, completion_texts2 = GOLDTrainer.generate_on_policy_outputs(
        trainer2,
        _DummyModelSmolLM(),
        {"prompts": batch["prompts"], "prompt_attention_mask": batch["prompt_attention_mask"]},
        generation_config2,
        pad_id2,
    )

    assert torch.equal(new_ids2, batch["input_ids"])
    if pad_id2 is not None:
        assert torch.equal(new_mask2, (batch["input_ids"] != pad_id2).long())
    else:
        assert torch.all(new_mask2 == 1)

    prompt_len = int(batch["prompt_attention_mask"].sum().item())
    tail_labels = new_labels2[0, prompt_len:]
    expected_tail = batch["input_ids"][0, prompt_len:]
    active_mask = tail_labels != -100
    assert torch.all(new_labels2[0, :prompt_len] == -100)
    assert torch.equal(tail_labels[active_mask], expected_tail[active_mask])
    assert torch.all(tail_labels[~active_mask] == -100)

    prompt_tokens = batch["prompts"][0, batch["prompt_attention_mask"][0].bool()]
    assert prompt_texts2[0] == smollm_tokenizer.decode(prompt_tokens.tolist(), skip_special_tokens=False)
    assert openr1_examples[0]["messages"][-1]["content"].strip() in completion_texts2[0]


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
