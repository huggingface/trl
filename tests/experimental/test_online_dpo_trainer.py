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

import pytest
import transformers
from datasets import Dataset, features, load_dataset
from packaging.version import Version
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import is_peft_available, is_vision_available

from trl.experimental.online_dpo import OnlineDPOConfig, OnlineDPOTrainer

from ..testing_utils import (
    RandomPairwiseJudge,
    TrlTestCase,
    require_llm_blender,
    require_peft,
    require_torch_accelerator,
    require_vision,
    require_vllm,
)


if is_peft_available():
    from peft import LoraConfig, get_peft_model

if is_vision_available():
    import numpy as np
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor


class TestOnlineDPOTrainer(TrlTestCase):
    def setup_method(self):
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.ref_model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.reward_model_id = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(self.reward_model_id, num_labels=1)
        self.reward_tokenizer = AutoTokenizer.from_pretrained(self.reward_model_id)
        self.reward_tokenizer.pad_token = self.reward_tokenizer.eos_token

    @pytest.mark.parametrize("config_name", ["standard_prompt_only", "conversational_prompt_only"])
    def test_training(self, config_name):
        training_args = OnlineDPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            learning_rate=5.0e-7,
            eval_strategy="steps",
            report_to="none",
        )
        dummy_dataset = load_dataset("trl-internal-testing/zen", config_name)

        trainer = OnlineDPOTrainer(
            model=self.model,
            reward_funcs=self.reward_model,
            args=training_args,
            train_dataset=dummy_dataset["train"],
            eval_dataset=dummy_dataset["test"],
            processing_class=self.tokenizer,
            reward_processing_classes=self.reward_tokenizer,
        )
        trainer.train()

        # Check if training loss is available
        assert "train_loss" in trainer.state.log_history[-1]

    def test_training_model_str(self):
        training_args = OnlineDPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            learning_rate=5.0e-7,
            eval_strategy="steps",
            report_to="none",
        )
        dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

        trainer = OnlineDPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=self.reward_model,
            args=training_args,
            train_dataset=dummy_dataset["train"],
            eval_dataset=dummy_dataset["test"],
            processing_class=self.tokenizer,
            reward_processing_classes=self.reward_tokenizer,
        )
        trainer.train()

        # Check if training loss is available
        assert "train_loss" in trainer.state.log_history[-1]

    def test_training_with_ref_model(self):
        training_args = OnlineDPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            learning_rate=5.0e-7,
            eval_strategy="steps",
            report_to="none",
        )
        dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

        trainer = OnlineDPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            reward_funcs=self.reward_model,
            args=training_args,
            train_dataset=dummy_dataset["train"],
            eval_dataset=dummy_dataset["test"],
            processing_class=self.tokenizer,
            reward_processing_classes=self.reward_tokenizer,
        )
        trainer.train()

        # Check if training loss is available
        assert "train_loss" in trainer.state.log_history[-1]

    def test_ref_model_is_model(self):
        training_args = OnlineDPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            report_to="none",
        )

        dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

        with pytest.raises(ValueError):
            OnlineDPOTrainer(
                model=self.model,
                ref_model=self.model,  # ref_model can't be the same as model
                reward_funcs=self.reward_model,
                args=training_args,
                train_dataset=dummy_dataset["train"],
                processing_class=self.tokenizer,
                reward_processing_classes=self.reward_tokenizer,
            )

    @require_peft
    def test_training_with_peft(self):
        lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        training_args = OnlineDPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            learning_rate=5.0e-7,
            eval_strategy="steps",
            report_to="none",
        )
        dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

        trainer = OnlineDPOTrainer(
            model=self.model,
            reward_funcs=self.reward_model,
            args=training_args,
            train_dataset=dummy_dataset["train"],
            eval_dataset=dummy_dataset["test"],
            processing_class=self.tokenizer,
            reward_processing_classes=self.reward_tokenizer,
            peft_config=lora_config,
        )

        trainer.train()

        # Check if training loss is available
        assert "train_loss" in trainer.state.log_history[-1]

    @require_peft
    def test_training_with_peft_and_ref_model(self):
        lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        training_args = OnlineDPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            learning_rate=5.0e-7,
            eval_strategy="steps",
            report_to="none",
        )
        dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

        trainer = OnlineDPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            reward_funcs=self.reward_model,
            args=training_args,
            train_dataset=dummy_dataset["train"],
            eval_dataset=dummy_dataset["test"],
            processing_class=self.tokenizer,
            reward_processing_classes=self.reward_tokenizer,
            peft_config=lora_config,
        )

        trainer.train()

        # Check if training loss is available
        assert "train_loss" in trainer.state.log_history[-1]

    @require_peft
    def test_training_with_peft_model_and_peft_config(self):
        model_lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(self.model, model_lora_config)
        # we want only the "train adapter" to be trained
        lora_train_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        training_args = OnlineDPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            learning_rate=5.0e-7,
            eval_strategy="steps",
            report_to="none",
        )
        dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

        trainer = OnlineDPOTrainer(
            model=model,
            reward_funcs=self.reward_model,
            args=training_args,
            train_dataset=dummy_dataset["train"],
            eval_dataset=dummy_dataset["test"],
            processing_class=self.tokenizer,
            reward_processing_classes=self.reward_tokenizer,
            peft_config=lora_train_config,
        )

        trainer.train()

        # Check if training loss is available
        assert "train_loss" in trainer.state.log_history[-1]

    @pytest.mark.parametrize("config_name", ["standard_prompt_only", "conversational_prompt_only"])
    @require_llm_blender
    def test_training_with_judge(self, config_name):
        training_args = OnlineDPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            learning_rate=5.0e-7,
            eval_strategy="steps",
            report_to="none",
        )
        dummy_dataset = load_dataset("trl-internal-testing/zen", config_name)

        trainer = OnlineDPOTrainer(
            model=self.model,
            judge=RandomPairwiseJudge(),
            args=training_args,
            train_dataset=dummy_dataset["train"],
            eval_dataset=dummy_dataset["test"],
            processing_class=self.tokenizer,
        )
        trainer.train()

        # Check if training loss is available
        assert "train_loss" in trainer.state.log_history[-1]

    @pytest.mark.parametrize("config_name", ["standard_prompt_only", "conversational_prompt_only"])
    @require_torch_accelerator
    @require_vllm
    @pytest.mark.slow
    def test_training_with_vllm(self, config_name):
        def cleanup_vllm_communicator(trainer):
            """Clean up vLLM communicator to avoid conflicts between test runs"""
            try:
                if hasattr(trainer, "vllm_client") and trainer.vllm_client is not None:
                    trainer.vllm_client.close_communicator()
            except Exception:
                pass  # Continue if cleanup fails

        model_id = "trl-internal-testing/small-Qwen2ForCausalLM-2.5"  # We need a bigger model
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        training_args = OnlineDPOConfig(
            output_dir=self.tmp_dir,
            use_vllm=True,
            vllm_gpu_memory_utilization=0.2,
            report_to="none",
        )
        dummy_dataset = load_dataset("trl-internal-testing/zen", config_name)

        trainer = OnlineDPOTrainer(
            model=model,
            reward_funcs=self.reward_model,
            args=training_args,
            train_dataset=dummy_dataset["train"],
            processing_class=tokenizer,
            reward_processing_classes=self.reward_tokenizer,
        )

        # Ensure cleanup of vLLM communicator after the test
        try:
            trainer.train()
            # Check if training loss is available
            assert "train_loss" in trainer.state.log_history[-1]
        finally:
            cleanup_vllm_communicator(trainer)

    @require_vllm
    def test_training_with_vllm_colocate(self):
        """Test vLLM colocate mode with our refactored implementation"""
        model_id = "trl-internal-testing/small-Qwen2ForCausalLM-2.5"  # We need a bigger model
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        training_args = OnlineDPOConfig(
            output_dir=self.tmp_dir,
            use_vllm=True,
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=0.2,
            per_device_train_batch_size=1,
            max_steps=2,
            report_to="none",
            # Test generation parameters
            temperature=0.9,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1,
            max_new_tokens=32,
        )
        dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

        trainer = OnlineDPOTrainer(
            model=model,
            reward_funcs=self.reward_model,
            args=training_args,
            train_dataset=dummy_dataset["train"],
            processing_class=tokenizer,
            reward_processing_classes=self.reward_tokenizer,
        )

        # Verify vLLM setup
        assert trainer.use_vllm
        assert trainer.vllm_mode == "colocate"
        assert trainer.llm is not None
        # self.assertIsNone(trainer.vllm_client)
        # self.assertEqual(trainer.vllm_gpu_memory_utilization, 0.2)

        # Verify generation parameters
        assert trainer.temperature == 0.9
        assert trainer.top_p == 0.95
        assert trainer.top_k == 50
        assert trainer.repetition_penalty == 1.1

        # Verify generation config
        assert trainer.generation_config is not None
        assert trainer.generation_config.temperature == 0.9
        assert trainer.generation_config.top_p == 0.95
        assert trainer.generation_config.top_k == 50
        assert trainer.generation_config.repetition_penalty == 1.1
        assert trainer.generation_config.max_tokens == 32

        trainer.train()

        # Check if training loss is available
        assert "train_loss" in trainer.state.log_history[-1]

    def test_vllm_config_validation(self):
        """Test vLLM configuration validation"""
        # Test valid vllm_mode values
        config = OnlineDPOConfig(use_vllm=True, vllm_mode="server")
        assert config.vllm_mode == "server"

        config = OnlineDPOConfig(use_vllm=True, vllm_mode="colocate")
        assert config.vllm_mode == "colocate"

        # Test default values
        config = OnlineDPOConfig()
        assert config.vllm_mode == "server"
        assert config.vllm_server_base_url is None
        assert config.vllm_server_host == "0.0.0.0"
        assert config.vllm_server_port == 8000
        assert config.vllm_server_timeout == 240.0
        assert config.vllm_gpu_memory_utilization == 0.55

        # Test generation parameters
        assert config.top_p == 1.0
        assert config.top_k is None
        assert config.min_p is None
        assert config.repetition_penalty == 1.0
        assert not config.use_transformers_paged
        assert config.cache_implementation is None
        assert config.generation_kwargs is None

    def test_generation_config_setup(self):
        """Test that generation configuration is properly set up for both vLLM and transformers"""
        training_args = OnlineDPOConfig(
            output_dir=self.tmp_dir,
            use_vllm=False,
            temperature=0.8,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.2,
            max_new_tokens=64,
            generation_kwargs={"do_sample": False},
            report_to="none",
        )
        dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

        trainer = OnlineDPOTrainer(
            model=self.model,
            reward_funcs=self.reward_model,
            args=training_args,
            train_dataset=dummy_dataset["train"],
            processing_class=self.tokenizer,
            reward_processing_classes=self.reward_tokenizer,
        )

        # Verify transformers generation config
        assert not trainer.use_vllm
        # When not using vLLM, these attributes should not be set
        assert not (hasattr(trainer, "llm") and trainer.llm is not None)
        assert not (hasattr(trainer, "vllm_client") and trainer.vllm_client is not None)
        assert trainer.generation_config is not None
        assert trainer.generation_config.temperature == 0.8
        assert trainer.generation_config.top_p == 0.9
        assert trainer.generation_config.top_k == 40
        assert trainer.generation_config.repetition_penalty == 1.2
        assert trainer.generation_config.max_new_tokens == 64
        assert not trainer.generation_config.do_sample  # From generation_kwargs

    @pytest.mark.parametrize("config_name", ["standard_prompt_only", "conversational_prompt_only"])
    @require_torch_accelerator
    def test_training_with_transformers_paged(self, config_name):
        if Version(transformers.__version__) < Version("4.57.0"):
            pytest.xfail("Upstream bug in transformers (GH#40692). Fix merged; awaiting release >= 4.57.0")
        training_args = OnlineDPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            learning_rate=5.0e-7,
            eval_strategy="steps",
            report_to="none",
            use_transformers_paged=True,
        )
        dummy_dataset = load_dataset("trl-internal-testing/zen", config_name)

        trainer = OnlineDPOTrainer(
            model=self.model,
            reward_funcs=self.reward_model,
            args=training_args,
            train_dataset=dummy_dataset["train"],
            eval_dataset=dummy_dataset["test"],
            processing_class=self.tokenizer,
            reward_processing_classes=self.reward_tokenizer,
        )
        trainer.train()

        # Check if training loss is available
        assert "train_loss" in trainer.state.log_history[-1]

    @pytest.mark.parametrize("config_name", ["standard_prompt_only", "conversational_prompt_only"])
    def test_training_with_reward_funcs(self, config_name):
        def simple_reward_func(prompts, completions, completion_ids, **kwargs):
            return [0.5 for _ in prompts]

        training_args = OnlineDPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            learning_rate=5.0e-7,
            eval_strategy="steps",
            reward_weights=[0.7, 0.3],
            report_to="none",
        )
        dummy_dataset = load_dataset("trl-internal-testing/zen", config_name)

        trainer = OnlineDPOTrainer(
            model=self.model,
            reward_funcs=[simple_reward_func, simple_reward_func],
            args=training_args,
            train_dataset=dummy_dataset["train"],
            eval_dataset=dummy_dataset["test"],
            processing_class=self.tokenizer,
        )
        trainer.train()

        assert "train_loss" in trainer.state.log_history[-1]
        assert len(trainer.reward_funcs) == 2
        assert trainer.reward_weights is not None
        assert round(abs(trainer.reward_weights[0].item() - 0.7), 5) == 0
        assert round(abs(trainer.reward_weights[1].item() - 0.3), 5) == 0


@require_vision
class TestOnlineDPOVisionTrainer(TrlTestCase):
    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Idefics2ForConditionalGeneration",
            "trl-internal-testing/tiny-LlavaForConditionalGeneration",
        ],
    )
    def test_online_dpo_vlm_trainer(self, model_id):
        dataset_dict = {
            "prompt": [
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the image."}]}],
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What do you see?"}]}],
            ],
            "images": [
                [Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))],
                [Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))],
            ],
        }
        dataset = Dataset.from_dict(dataset_dict)
        dataset = dataset.cast_column("images", features.Sequence(features.Image()))

        model = AutoModelForImageTextToText.from_pretrained(model_id)
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            "trl-internal-testing/tiny-LlamaForCausalLM-3.2", num_labels=1
        )
        processor = AutoProcessor.from_pretrained(model_id)
        reward_tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-LlamaForCausalLM-3.2")
        reward_tokenizer.pad_token = reward_tokenizer.eos_token

        training_args = OnlineDPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=1,
            max_steps=2,
            learning_rate=0.01,
            report_to="none",
        )
        trainer = OnlineDPOTrainer(
            model=model,
            reward_funcs=reward_model,
            args=training_args,
            processing_class=processor,
            train_dataset=dataset,
            eval_dataset=dataset,
            reward_processing_classes=reward_tokenizer,
        )

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
