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

import pytest
import torch
from accelerate import Accelerator
from datasets import DatasetDict, load_dataset
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers.utils import is_peft_available

from trl.experimental.bco import BCOConfig, BCOTrainer
from trl.experimental.bco.bco_trainer import _process_tokens, _tokenize

from ..testing_utils import TrlTestCase, require_no_wandb, require_peft, require_sklearn


if is_peft_available():
    from peft import LoraConfig


@pytest.mark.low_priority
class TestBCOTrainer(TrlTestCase):
    @require_sklearn
    def test_trust_remote_code(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")
        model_id = "trl-internal-testing/tiny-RemoteForCausalLM"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        with pytest.raises(ValueError, match="custom code"):
            BCOTrainer(
                model=model_id,
                args=BCOConfig(output_dir=self.tmp_dir, report_to="none"),
                processing_class=tokenizer,
                train_dataset=dataset,
            )

        trainer = BCOTrainer(
            model=model_id,
            args=BCOConfig(output_dir=self.tmp_dir, report_to="none", trust_remote_code=True),
            processing_class=tokenizer,
            train_dataset=dataset,
        )
        assert type(trainer.model).__name__ == "RemoteForCausalLM"

    @pytest.mark.parametrize(
        "config_name",
        [
            "standard_preference",
            "standard_implicit_prompt_preference",
            "standard_unpaired_preference",
            "conversational_preference",
            "conversational_implicit_prompt_preference",
            "conversational_unpaired_preference",
        ],
    )
    @require_sklearn
    def test_train(self, config_name):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        ref_model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        dataset = load_dataset("trl-internal-testing/zen", config_name, split="train")

        training_args = BCOConfig(
            output_dir=self.tmp_dir,
            remove_unused_columns=False,  # warning raised if not set to False
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )

        trainer = BCOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if param.sum() != 0:  # ignore 0 biases
                assert not torch.equal(param.cpu(), new_param.cpu())

    @pytest.mark.parametrize(
        "eval_dataset_type",
        [
            "dataset",
            "dataset_dict",
            "none",
        ],
    )
    @require_sklearn
    def test_init_with_eval_dataset(self, eval_dataset_type):
        # BCO tokenizes the eval dataset at init by calling `.map()` directly on it, so streaming (iterable) and
        # plain dict-of-datasets eval datasets are not yet supported; only `Dataset` and `DatasetDict` are tested.
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        ref_model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        train_dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")

        if eval_dataset_type == "none":
            eval_dataset = None
        else:
            eval_split = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="test")
            if eval_dataset_type == "dataset":
                eval_dataset = eval_split
            else:  # "dataset_dict"
                eval_dataset = DatasetDict({"data1": eval_split, "data2": eval_split})

        training_args = BCOConfig(output_dir=self.tmp_dir, remove_unused_columns=False, report_to="none")
        trainer = BCOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        if eval_dataset_type == "none":
            assert trainer.eval_dataset is None
        elif isinstance(trainer.eval_dataset, dict):
            assert set(trainer.eval_dataset.keys()) == {"data1", "data2"}
            # Each split was tokenized independently.
            assert "prompt_input_ids" in next(iter(trainer.eval_dataset["data1"]))
            assert "prompt_input_ids" in next(iter(trainer.eval_dataset["data2"]))
        else:
            assert "prompt_input_ids" in next(iter(trainer.eval_dataset))

    @require_sklearn
    def test_train_processing_class_autoloaded(self):
        # processing_class is documented as optional: when omitted it should be
        # auto-loaded from the model, consistent with DPOTrainer / RewardTrainer.
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        ref_model = AutoModelForCausalLM.from_pretrained(model_id)
        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")
        training_args = BCOConfig(
            output_dir=self.tmp_dir,
            remove_unused_columns=False,
            learning_rate=0.1,
            report_to="none",
        )
        trainer = BCOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=dataset,
        )
        assert trainer.processing_class is not None
        trainer.train()
        assert trainer.state.log_history[-1]["train_loss"] is not None

    @require_sklearn
    def test_train_with_precompute(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        ref_model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")

        training_args = BCOConfig(
            output_dir=self.tmp_dir,
            remove_unused_columns=False,  # warning raised if not set to False
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            precompute_ref_log_probs=True,
            report_to="none",
        )

        trainer = BCOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if param.sum() != 0:  # ignore 0 biases
                assert not torch.equal(param.cpu(), new_param.cpu())

    @require_sklearn
    def test_train_eval(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        ref_model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference")

        training_args = BCOConfig(
            output_dir=self.tmp_dir,
            remove_unused_columns=False,  # warning raised if not set to False
            eval_strategy="steps",
            eval_steps=3,
            report_to="none",
        )

        trainer = BCOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
        )

        trainer.train()

    @require_sklearn
    def test_init_with_ref_model_is_model(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")

        training_args = BCOConfig(
            output_dir=self.tmp_dir,
            remove_unused_columns=False,  # warning raised if not set to False
            report_to="none",
        )

        with pytest.raises(ValueError):
            BCOTrainer(
                model=model,
                ref_model=model,  # ref_model can't be the same as model
                args=training_args,
                processing_class=tokenizer,
                train_dataset=dataset,
            )

    @require_sklearn
    def test_tokenize_and_process_tokens(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        ref_model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")

        training_args = BCOConfig(
            output_dir=self.tmp_dir,
            remove_unused_columns=False,  # warning raised if not set to False
            report_to="none",
        )

        trainer = BCOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=dataset,
        )

        tokenized_dataset = dataset.map(
            _tokenize,
            fn_kwargs={"tokenizer": trainer.processing_class},
            batched=True,
            batch_size=2,
        )
        assert tokenized_dataset["prompt"][:] == dataset["prompt"][:]
        assert tokenized_dataset["completion"][:] == dataset["completion"][:]
        assert tokenized_dataset["label"][:] == dataset["label"][:]
        assert tokenized_dataset["prompt_input_ids"][0] == [46518, 374, 2664, 1091]
        assert tokenized_dataset["prompt_attention_mask"][0] == [1, 1, 1, 1]
        assert tokenized_dataset["answer_input_ids"][0] == [27261, 13]
        assert tokenized_dataset["answer_attention_mask"][0] == [1, 1]

        fn_kwargs = {
            "prefix": "",
            "is_encoder_decoder": trainer.is_encoder_decoder,
            "tokenizer": trainer.processing_class,
            "max_length": trainer.max_length,
        }
        processed_dataset = tokenized_dataset.map(_process_tokens, fn_kwargs=fn_kwargs)
        assert processed_dataset["prompt"][:] == dataset["prompt"][:]
        assert processed_dataset["completion"][:] == dataset["completion"][:]
        assert processed_dataset["label"][:] == dataset["label"][:]
        assert processed_dataset["prompt_input_ids"][0] == [46518, 374, 2664, 1091]
        assert processed_dataset["prompt_attention_mask"][0] == [1, 1, 1, 1]
        assert processed_dataset["completion_input_ids"][0] == [46518, 374, 2664, 1091, 27261, 13, 151645]
        assert processed_dataset["completion_attention_mask"][0] == [1, 1, 1, 1, 1, 1, 1]
        assert processed_dataset["completion_labels"][0] == [-100, -100, -100, -100, 27261, 13, 151645]

    @pytest.mark.parametrize(
        ("prompt_length", "answer_length"),
        [
            (4, 6),  # everything fits, nothing is truncated
            (28, 10),  # answer truncated, budget still positive
            (31, 1),  # prompt one token short of max_length: bound goes negative by one
            (60, 10),  # prompt alone exceeds max_length
            (1000, 10),  # prompt exceeds max_length by a lot
        ],
    )
    def test_process_tokens_truncation_respects_max_length(self, prompt_length, answer_length):
        """`_process_tokens` must keep the sequence within `max_length` and only ever truncate prefixes.

        The response budget is `max_length` minus the prompt, so a prompt longer than `max_length` used to make
        that bound negative. `answer[:negative]` slices from the *end* of the answer, which empties it outright
        whenever the answer is shorter than the overflow, and the resulting sequence stayed over `max_length`
        because the prompt was never truncated.
        """
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        max_length = 32
        # Synthetic ids: the branch under test only looks at lengths and at the BOS/EOS ids.
        prompt_input_ids = list(range(100, 100 + prompt_length))
        answer_input_ids = list(range(200, 200 + answer_length))
        example = {
            "prompt": "prompt",
            "completion": "completion",
            "label": True,
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": [1] * prompt_length,
            "answer_input_ids": answer_input_ids,
            "answer_attention_mask": [1] * answer_length,
        }

        processed = _process_tokens(
            example,
            prefix="",
            is_encoder_decoder=False,
            tokenizer=tokenizer,
            max_length=max_length,
        )

        completion_input_ids = processed["completion_input_ids"]
        assert len(completion_input_ids) <= max_length
        assert len(processed["completion_attention_mask"]) == len(completion_input_ids)
        assert len(processed["completion_labels"]) == len(completion_input_ids)

        # Both halves must stay prefixes of what went in. A negative slice bound cuts from the end instead.
        kept_prompt = processed["prompt_input_ids"]
        if tokenizer.bos_token_id is not None and kept_prompt and kept_prompt[0] == tokenizer.bos_token_id:
            kept_prompt = kept_prompt[1:]
        assert kept_prompt == prompt_input_ids[: len(kept_prompt)]

        kept_answer = completion_input_ids[len(processed["prompt_input_ids"]) :]
        if kept_answer and kept_answer[-1] == tokenizer.eos_token_id:
            kept_answer = kept_answer[:-1]
        assert kept_answer == answer_input_ids[: len(kept_answer)]

        # The completion must survive as long as the prompt leaves room for it.
        if prompt_length + answer_length <= max_length - 2:
            assert kept_answer == answer_input_ids

    @require_sklearn
    def test_train_without_providing_ref_model(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")

        training_args = BCOConfig(
            output_dir=self.tmp_dir,
            remove_unused_columns=False,  # warning raised if not set to False
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )

        trainer = BCOTrainer(
            model=model,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if param.sum() != 0:  # ignore 0 biases
                assert not torch.equal(param.cpu(), new_param.cpu())

    @require_sklearn
    def test_train_udm(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Get embedding model
        embedding_model_id = "trl-internal-testing/tiny-BartModel"
        embedding_model = AutoModel.from_pretrained(embedding_model_id)
        embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_id)

        def embed_prompt(input_ids, attention_mask, model):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            return outputs.last_hidden_state.mean(dim=1)

        embedding_model = Accelerator().prepare_model(embedding_model)
        embedding_func = partial(embed_prompt, model=embedding_model)

        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")

        training_args = BCOConfig(
            output_dir=self.tmp_dir,
            remove_unused_columns=False,  # warning raised if not set to False
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )

        trainer = BCOTrainer(
            model=model,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=dataset,
            embedding_func=embedding_func,
            embedding_tokenizer=embedding_tokenizer,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if param.sum() != 0:  # ignore 0 biases
                assert not torch.equal(param.cpu(), new_param.cpu())

    @require_sklearn
    @require_peft
    def test_train_without_providing_ref_model_with_lora(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")

        training_args = BCOConfig(
            output_dir=self.tmp_dir,
            remove_unused_columns=False,  # warning raised if not set to False
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )

        trainer = BCOTrainer(
            model=model,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=dataset,
            peft_config=lora_config,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            if "lora" in n:
                new_param = trainer.model.get_parameter(n)
                if param.sum() != 0:  # ignore 0 biases
                    assert not torch.equal(param.cpu(), new_param.cpu())

    @require_sklearn
    @require_no_wandb
    def test_generate_during_eval_no_wandb(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference")

        training_args = BCOConfig(
            output_dir=self.tmp_dir,
            remove_unused_columns=False,  # warning raised if not set to False
            eval_strategy="steps",
            eval_steps=3,
            generate_during_eval=True,
            report_to="none",
        )

        with pytest.raises(
            ValueError,
            match="`generate_during_eval=True` requires Weights and Biases or Comet to be installed."
            " Please install `wandb` or `comet-ml` to resolve.",
        ):
            BCOTrainer(
                model=model,
                args=training_args,
                processing_class=tokenizer,
                train_dataset=dataset["train"],
                eval_dataset=dataset["test"],
            )

    @require_sklearn
    @require_peft
    def test_lora_train_and_save(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")

        training_args = BCOConfig(
            output_dir=self.tmp_dir,
            remove_unused_columns=False,  # warning raised if not set to False
            report_to="none",
        )

        trainer = BCOTrainer(
            model=model,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=dataset,
            peft_config=lora_config,
        )

        # train the model
        trainer.train()

        # save peft adapter
        trainer.save_model()

        # assert that the model is loaded without giving OSError
        AutoModelForCausalLM.from_pretrained(self.tmp_dir)

    @require_sklearn
    def test_compute_metrics(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        ref_model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference")

        def dummy_compute_metrics(*args, **kwargs):
            return {"test": 0.0}

        training_args = BCOConfig(
            output_dir=self.tmp_dir,
            remove_unused_columns=False,  # warning raised if not set to False
            eval_strategy="steps",
            eval_steps=3,
            report_to="none",
        )

        trainer = BCOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            compute_metrics=dummy_compute_metrics,
        )

        trainer.train()

        assert trainer.state.log_history[-2]["eval_test"] == 0.0
