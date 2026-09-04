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

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset, DatasetDict, features, load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import is_peft_available, is_vision_available

from trl.experimental.online_dpo import OnlineDPOConfig, OnlineDPOTrainer

from ..testing_utils import (
    TrlTestCase,
    drop_last_for_metrics,
    require_peft,
    require_torch_accelerator,
    require_vision,
    require_vllm,
)


if is_peft_available():
    from peft import LoraConfig


if is_vision_available():
    import numpy as np
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor


def test_objective_metrics_ignore_duplicate_padding():
    batch_size = 3
    model = torch.nn.Linear(1, 1)
    ref_model = torch.nn.Linear(1, 1)
    completion_ids = torch.ones(2 * batch_size, 2, dtype=torch.long)
    completion_mask = torch.tensor([[1, 1], [1, 0], [1, 0], [1, 1], [1, 0], [1, 0]])
    logprobs = torch.tensor(
        [[-1.0, -1.0], [-2.0, -2.0], [-30.0, -30.0], [-3.0, -3.0], [-4.0, -4.0], [-40.0, -40.0]],
        requires_grad=True,
    )
    ref_logprobs = torch.tensor(
        [[-2.0, -2.0], [-3.0, -3.0], [-10.0, -10.0], [-4.0, -4.0], [-5.0, -5.0], [-20.0, -20.0]]
    )
    rewards = torch.tensor([3.0, 4.0, 100.0, 1.0, 2.0, -100.0])

    def forward(current_model, *args):
        return logprobs if current_model is model else ref_logprobs

    trainer = SimpleNamespace(
        args=SimpleNamespace(
            use_vllm=False,
            missing_eos_penalty=None,
            loss_type="sigmoid",
            torch_empty_cache_steps=None,
            optim="adamw_torch",
            n_gpu=1,
            gradient_accumulation_steps=1,
        ),
        ref_model=ref_model,
        model=model,
        reward_funcs=[object()],
        processing_class=SimpleNamespace(
            eos_token_id=1,
            batch_decode=lambda token_ids, skip_special_tokens: ["completion"] * len(token_ids),
        ),
        _tokenizer=SimpleNamespace(eos_token_id=1),
        beta=0.5,
        stats=defaultdict(list),
        state=SimpleNamespace(global_step=1),
        accelerator=SimpleNamespace(gather_for_metrics=drop_last_for_metrics, backward=lambda loss, **kwargs: None),
        _generate=lambda current_model, prompts, images: (
            torch.ones(2 * batch_size, 1, dtype=torch.long),
            torch.ones(2 * batch_size, 1, dtype=torch.long),
            completion_ids,
            completion_mask,
        ),
        _forward=forward,
        _calculate_rewards_from_functions=lambda **kwargs: rewards,
    )

    OnlineDPOTrainer.training_step(trainer, model, {"prompt": ["a", "b", "outlier"]})

    kl_per_token = (logprobs - ref_logprobs) * completion_mask
    kl = kl_per_token.sum(1).view(2, batch_size).mean(0)
    non_score_reward = (-trainer.beta * kl_per_token).sum(1)
    expected = {
        "objective/scores_margin": rewards[:batch_size] - rewards[batch_size:],
        "objective/scores": rewards.view(2, batch_size).mean(0),
        "objective/kl": kl,
        "objective/non_score_reward": non_score_reward.view(2, batch_size).mean(0),
        "objective/rlhf_reward": (rewards + non_score_reward).view(2, batch_size).mean(0),
        "objective/entropy": -(logprobs * completion_mask).sum(1).view(2, batch_size).mean(0),
    }
    for key, values in expected.items():
        assert trainer.stats[key][-1] == values[:-1].mean().item()


class TestOnlineDPOTrainer(TrlTestCase):
    def setup_method(self):
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype="float32")
        self.ref_model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.reward_model_id = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(self.reward_model_id, num_labels=1)
        self.reward_tokenizer = AutoTokenizer.from_pretrained(self.reward_model_id)
        self.reward_tokenizer.pad_token = self.reward_tokenizer.eos_token

    def test_trust_remote_code(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        model_id = "trl-internal-testing/tiny-RemoteForCausalLM"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        with pytest.raises(ValueError, match="custom code"):
            OnlineDPOTrainer(
                model=model_id,
                reward_funcs=self.reward_model,
                args=OnlineDPOConfig(output_dir=self.tmp_dir, report_to="none"),
                train_dataset=dataset,
                processing_class=tokenizer,
                reward_processing_classes=self.reward_tokenizer,
            )

        trainer = OnlineDPOTrainer(
            model=model_id,
            reward_funcs=self.reward_model,
            args=OnlineDPOConfig(output_dir=self.tmp_dir, report_to="none", trust_remote_code=True),
            train_dataset=dataset,
            processing_class=tokenizer,
            reward_processing_classes=self.reward_tokenizer,
        )
        assert type(trainer.model).__name__ == "RemoteForCausalLM"

    @pytest.mark.parametrize("config_name", ["standard_prompt_only", "conversational_prompt_only"])
    def test_train(self, config_name):
        training_args = OnlineDPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            learning_rate=5.0e-7,
            report_to="none",
        )
        dataset = load_dataset("trl-internal-testing/zen", config_name, split="train")

        trainer = OnlineDPOTrainer(
            model=self.model,
            reward_funcs=self.reward_model,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
            reward_processing_classes=self.reward_tokenizer,
        )
        trainer.train()

        assert "train_loss" in trainer.state.log_history[-1]

    @pytest.mark.parametrize("eval_dataset_type", ["dataset", "dataset_dict", "dict_of_dataset", "none"])
    def test_init_with_eval_dataset(self, eval_dataset_type):
        # Streaming datasets are not yet supported in OnlineDPO
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

        if eval_dataset_type == "none":
            eval_dataset = None
        elif eval_dataset_type == "dataset":
            eval_dataset = dataset["test"]
        elif eval_dataset_type == "dataset_dict":
            eval_dataset = DatasetDict({"data1": dataset["test"], "data2": dataset["test"]})
        else:  # "dict_of_dataset"
            eval_dataset = {"data1": dataset["test"], "data2": dataset["test"]}

        training_args = OnlineDPOConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = OnlineDPOTrainer(
            model=self.model,
            reward_funcs=self.reward_model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            reward_processing_classes=self.reward_tokenizer,
        )

        if eval_dataset_type == "none":
            assert trainer.eval_dataset is None
        elif isinstance(trainer.eval_dataset, dict):
            assert set(trainer.eval_dataset.keys()) == {"data1", "data2"}
        else:
            assert trainer.eval_dataset is eval_dataset

    def test_train_model_str(self):
        training_args = OnlineDPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            learning_rate=5.0e-7,
            report_to="none",
        )
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        trainer = OnlineDPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=self.reward_model,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
            reward_processing_classes=self.reward_tokenizer,
        )
        trainer.train()

        assert "train_loss" in trainer.state.log_history[-1]

    def test_train_with_ref_model(self):
        training_args = OnlineDPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            learning_rate=5.0e-7,
            report_to="none",
        )
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        trainer = OnlineDPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            reward_funcs=self.reward_model,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
            reward_processing_classes=self.reward_tokenizer,
        )
        trainer.train()

        assert "train_loss" in trainer.state.log_history[-1]

    def test_ref_model_is_model(self):
        training_args = OnlineDPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            report_to="none",
        )

        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with pytest.raises(ValueError):
            OnlineDPOTrainer(
                model=self.model,
                ref_model=self.model,  # ref_model can't be the same as model
                reward_funcs=self.reward_model,
                args=training_args,
                train_dataset=dataset,
                processing_class=self.tokenizer,
                reward_processing_classes=self.reward_tokenizer,
            )

    @require_peft
    def test_train_with_peft(self):
        lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        training_args = OnlineDPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            learning_rate=5.0e-7,
            report_to="none",
        )
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        trainer = OnlineDPOTrainer(
            model=self.model,
            reward_funcs=self.reward_model,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
            reward_processing_classes=self.reward_tokenizer,
            peft_config=lora_config,
        )

        trainer.train()

        assert "train_loss" in trainer.state.log_history[-1]

    @require_peft
    def test_train_with_peft_and_ref_model(self):
        lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        training_args = OnlineDPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            learning_rate=5.0e-7,
            report_to="none",
        )
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        trainer = OnlineDPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            reward_funcs=self.reward_model,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
            reward_processing_classes=self.reward_tokenizer,
            peft_config=lora_config,
        )

        trainer.train()

        assert "train_loss" in trainer.state.log_history[-1]

    @pytest.mark.parametrize("config_name", ["standard_prompt_only", "conversational_prompt_only"])
    @require_torch_accelerator
    @require_vllm
    @pytest.mark.slow
    def test_train_with_vllm_server(self, config_name):
        def cleanup_vllm_communicator(trainer):
            """Clean up vLLM communicator to avoid conflicts between test runs"""
            try:
                if hasattr(trainer, "vllm_client") and trainer.vllm_client is not None:
                    trainer.vllm_client.close_communicator()
            except Exception:
                pass  # Continue if cleanup fails

        model_id = "trl-internal-testing/small-Qwen2ForCausalLM-2.5"  # We need a bigger model
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        training_args = OnlineDPOConfig(
            output_dir=self.tmp_dir,
            use_vllm=True,
            vllm_mode="server",
            vllm_gpu_memory_utilization=0.2,
            report_to="none",
        )
        dataset = load_dataset("trl-internal-testing/zen", config_name, split="train")

        trainer = OnlineDPOTrainer(
            model=model,
            reward_funcs=self.reward_model,
            args=training_args,
            train_dataset=dataset,
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
    def test_train_with_vllm_colocate(self):
        """Test vLLM colocate mode with our refactored implementation"""
        model_id = "trl-internal-testing/small-Qwen2ForCausalLM-2.5"  # We need a bigger model
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32")
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
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        trainer = OnlineDPOTrainer(
            model=model,
            reward_funcs=self.reward_model,
            args=training_args,
            train_dataset=dataset,
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
        assert config.vllm_mode == "colocate"
        assert config.vllm_server_base_url is None
        assert config.vllm_server_host == "0.0.0.0"
        assert config.vllm_server_port == 8000
        assert config.vllm_server_timeout == 240.0
        assert config.vllm_gpu_memory_utilization == 0.55

        # Test generation parameters
        assert config.top_p == 1.0
        assert config.top_k == 0
        assert config.min_p is None
        assert config.repetition_penalty == 1.0
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
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        trainer = OnlineDPOTrainer(
            model=self.model,
            reward_funcs=self.reward_model,
            args=training_args,
            train_dataset=dataset,
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
    def test_train_with_reward_funcs(self, config_name):
        def simple_reward_func(prompts, completions, completion_ids, **kwargs):
            return [0.5 for _ in prompts]

        training_args = OnlineDPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            learning_rate=5.0e-7,
            reward_weights=[0.7, 0.3],
            report_to="none",
        )
        dataset = load_dataset("trl-internal-testing/zen", config_name, split="train")

        trainer = OnlineDPOTrainer(
            model=self.model,
            reward_funcs=[simple_reward_func, simple_reward_func],
            args=training_args,
            train_dataset=dataset,
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

        model = AutoModelForImageTextToText.from_pretrained(model_id, dtype="float32")
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
