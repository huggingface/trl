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

import gc
import os
import warnings

import numpy as np
import pytest
import torch
import transformers
from accelerate.utils.memory import release_memory
from datasets import Dataset, Features, Image, Value, load_dataset
from packaging.version import Version
from parameterized import parameterized
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.testing_utils import (
    backend_empty_cache,
    require_bitsandbytes,
    require_flash_attn,
    require_liger_kernel,
    require_peft,
    require_torch_accelerator,
    torch_device,
)
from transformers.utils import is_peft_available

from trl import GRPOConfig, GRPOTrainer
from trl.trainer.utils import get_kbit_device_map

from ..testing_utils import TrlTestCase, require_vllm
from .testing_constants import MODELS_TO_TEST


if is_peft_available():
    from peft import LoraConfig, PeftModel


@pytest.mark.slow
@require_torch_accelerator
class GRPOTrainerSlowTester(TrlTestCase):
    def setUp(self):
        super().setUp()
        self.train_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        self.eval_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="test")
        self.max_length = 128

    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)
        gc.collect()
        super().tearDown()

    @parameterized.expand(MODELS_TO_TEST)
    @require_liger_kernel
    def test_training_with_liger_grpo_loss(self, model_name):
        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,
            num_generations=3,
            use_liger_loss=True,
            max_completion_length=self.max_length,
            report_to="none",
            logging_strategy="no",
        )

        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

        trainer = GRPOTrainer(
            model=model,
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=tokenizer,
        )
        from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss

        assert isinstance(trainer.liger_grpo_loss, LigerFusedLinearGRPOLoss)

        previous_trainable_params = {n: param.clone() for n, param in model.named_parameters()}

        trainer.train()

        for n, param in previous_trainable_params.items():
            new_param = model.get_parameter(n)
            self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

        release_memory(model, trainer)

    @parameterized.expand(MODELS_TO_TEST)
    @require_liger_kernel
    @require_peft
    def test_training_with_liger_grpo_loss_and_peft(self, model_name):
        from peft import LoraConfig, TaskType

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,
            num_generations=3,
            use_liger_loss=True,
            max_completion_length=self.max_length,
            report_to="none",
            logging_strategy="no",
        )

        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

        # Configure PEFT with LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
        )

        trainer = GRPOTrainer(
            model=model,
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
        from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss

        assert isinstance(trainer.liger_grpo_loss, LigerFusedLinearGRPOLoss)

        # Verify PEFT adapter is properly initialized
        from peft import PeftModel

        self.assertTrue(isinstance(trainer.model, PeftModel), "Model should be wrapped with PEFT")

        # Store adapter weights before training
        previous_trainable_params = {
            n: param.clone() for n, param in trainer.model.named_parameters() if param.requires_grad
        }
        self.assertTrue(len(previous_trainable_params) > 0, "No trainable parameters found in PEFT model")

        trainer.train()

        # Verify adapter weights have changed after training
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

        release_memory(model, trainer)

    @parameterized.expand(MODELS_TO_TEST)
    def test_training_with_transformers_paged(self, model_name):
        """Test that training works with transformers paged implementation (requires GPU)."""
        if Version(transformers.__version__) < Version("4.57.0"):
            pytest.xfail("Upstream bug in transformers (GH#40692). Fix merged; awaiting release >= 4.57.0")
        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            use_transformers_paged=True,  # Enable transformers paged implementation
            report_to="none",
            logging_strategy="no",
        )

        model = AutoModelForCausalLM.from_pretrained(model_name)

        trainer = GRPOTrainer(
            model=model,
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=self.train_dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in model.named_parameters()}

        trainer.train()

        self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = model.get_parameter(n)
            self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

        release_memory(model, trainer)

    @require_flash_attn
    @require_bitsandbytes
    @require_peft
    @parameterized.expand(
        [
            ("HuggingFaceTB/SmolVLM-Instruct",),  # Only test the smaller model to avoid OOM
        ]
    )
    def test_vlm_training(self, model_name):
        """
        Test VLM training with aggressive memory optimization.

        This test uses multiple memory reduction techniques:
        - 4-bit quantization with double quantization
        - LoRA with very low rank (r=4)
        - Minimal batch size (1) with gradient accumulation
        - Small images (64x64 instead of 224x224)
        - Short sequences (max_completion_length=8)
        - Only 4 training samples
        - Only 1 training step
        - Gradient checkpointing and bfloat16
        """

        # Create processor once outside the data generator
        processor = AutoProcessor.from_pretrained(model_name, use_fast=True, padding_side="left")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is in the image?"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        def data_gen(num_samples):
            for _ in range(num_samples):
                yield {
                    "prompt": prompt,
                    "image": np.random.uniform(low=0.0, high=255.0, size=(64, 64, 3)).astype(
                        np.uint8
                    ),  # Much smaller images
                }

        dataset = Dataset.from_generator(
            data_gen, gen_kwargs={"num_samples": 4}, features=Features(image=Image(), prompt=Value(dtype="string"))
        )
        # reduce memory requirements as much as possible
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage="bfloat16",
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2",
            dtype="bfloat16",
            device_map=get_kbit_device_map(),
            quantization_config=quantization_config,
        )

        def reward_func(prompts, completions, **kwargs):
            # simple nonsensical reward
            return [-((len(c) - 25) ** 2) + 100 for c in completions]

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,  # Minimal batch size
            gradient_accumulation_steps=2,  # Maintain effective batch size
            num_generations=2,
            max_completion_length=8,  # Much shorter completions
            max_prompt_length=None,  # Don't limit prompt length for VLM
            bf16=True,  # Use bfloat16 precision
            max_steps=1,  # Only do 1 training step to save time and memory
            report_to="none",
            logging_strategy="no",
        )
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=4,  # Much lower rank for minimal memory
            lora_alpha=8,  # Reduced alpha proportionally
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],  # Minimal target modules
            # For VLM models, we typically want to freeze the vision encoder
            # and only adapt the language model parameters
            modules_to_save=None,
        )

        try:
            trainer = GRPOTrainer(
                model=model,
                processing_class=processor,
                reward_funcs=[reward_func],
                args=training_args,
                train_dataset=dataset,
                peft_config=lora_config,
            )

            self.assertIsInstance(trainer.model, PeftModel)

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that LoRA parameters have changed
            # For VLM models, we're more permissive about which parameters can change
            lora_params_changed = False
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                if "lora" in n.lower():  # LoRA parameters should change
                    if not torch.equal(param, new_param):
                        lora_params_changed = True

            # At least some LoRA parameters should have changed during training
            self.assertTrue(lora_params_changed, "No LoRA parameters were updated during training.")

        except torch.OutOfMemoryError as e:
            self.skipTest(f"Skipping VLM training test due to insufficient GPU memory: {e}")
        except Exception as e:
            # Check for other memory-related errors
            if any(keyword in str(e).lower() for keyword in ["memory", "cuda", "out of memory", "insufficient"]):
                self.skipTest(f"Skipping VLM training test due to hardware constraints: {e}")
            else:
                raise

        release_memory(model, trainer)

    @require_vllm
    @require_bitsandbytes
    @require_peft
    def test_vlm_processor_vllm_colocate_mode(self):
        """
        Test that VLM processors work with vLLM in colocate mode.

        This test uses multiple memory optimization techniques to ensure it runs on limited hardware:
        - LoRA (Low-Rank Adaptation) with minimal rank (r=4)
        - 4-bit quantization with BitsAndBytesConfig
        - Gradient checkpointing
        - bfloat16 precision
        - Minimal batch sizes and sequence lengths
        - Very low GPU memory utilization (5%)
        """
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        config = GRPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=1,  # Minimal batch size
            gradient_accumulation_steps=2,  # Make effective batch size 2, divisible by num_generations
            num_generations=2,
            max_completion_length=4,  # Very short completions to reduce memory
            max_prompt_length=32,  # Very short prompts to reduce memory
            use_vllm=True,  # Enable vLLM
            vllm_mode="colocate",  # Use colocate mode to avoid server dependency
            vllm_gpu_memory_utilization=0.05,  # Use minimal GPU memory (5%)
            gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
            bf16=True,  # Use bfloat16 to reduce memory
            report_to="none",
            logging_strategy="no",
        )

        # Create a VLM processor
        processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct", use_fast=True, padding_side="left")

        # Verify processor has both required attributes for VLM detection
        self.assertTrue(hasattr(processor, "tokenizer"))
        self.assertTrue(hasattr(processor, "image_processor"))

        def dummy_reward_func(completions, **kwargs):
            return [1.0] * len(completions)

        # Use LoRA configuration for memory efficiency
        lora_config = LoraConfig(
            r=4,  # Very low rank for minimal memory
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],  # Minimal target modules
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Use 4-bit quantization for further memory reduction
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        original_env = {}
        required_env_vars = {
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "LOCAL_WORLD_SIZE": "1",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12355",
        }

        for key, value in required_env_vars.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value

        try:
            # Test VLM processor with vLLM colocate mode
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    # Load model with quantization for memory efficiency
                    model = AutoModelForCausalLM.from_pretrained(
                        "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                        quantization_config=quantization_config,
                        dtype=torch.bfloat16,
                    )

                    trainer = GRPOTrainer(
                        model=model,
                        reward_funcs=dummy_reward_func,
                        args=config,
                        train_dataset=dataset,
                        processing_class=processor,  # VLM processor
                        peft_config=lora_config,  # Use LoRA for memory efficiency
                    )

                    # Should detect VLM processor correctly and allow vLLM
                    self.assertTrue(trainer.use_vllm, "vLLM should be enabled for VLM processors in colocate mode")
                    self.assertEqual(trainer.vllm_mode, "colocate", "Should use colocate mode")

                    # Check if signature columns were set properly
                    if trainer._signature_columns is not None:
                        # Should include 'image' in signature columns for VLM processors
                        self.assertIn(
                            "image",
                            trainer._signature_columns,
                            "Should include 'image' in signature columns for VLM",
                        )

                    # Should not emit any warnings about VLM incompatibility
                    incompatibility_warnings = [
                        str(w_item.message)
                        for w_item in w
                        if "does not support VLMs" in str(w_item.message)
                        or "not compatible" in str(w_item.message).lower()
                    ]
                    self.assertEqual(
                        len(incompatibility_warnings),
                        0,
                        f"Should not emit VLM incompatibility warnings, but got: {incompatibility_warnings}",
                    )

                    # Test passes if we get this far without exceptions

                except Exception as e:
                    # If vLLM fails to initialize due to hardware constraints or other issues, that's expected
                    if any(
                        keyword in str(e).lower()
                        for keyword in [
                            "outofmemoryerror",
                            "cuda",
                            "memory",
                            "insufficient",
                            "no such device",
                            "free memory",
                            "gpu memory utilization",
                            "decrease gpu memory",
                        ]
                    ):
                        self.skipTest(f"Skipping vLLM colocate test due to hardware constraints: {e}")
                    elif "KeyError" in str(e) and "RANK" in str(e):
                        self.skipTest(f"Skipping vLLM colocate test due to environment setup issues: {e}")
                    elif "ValueError" in str(e) and "memory" in str(e).lower():
                        self.skipTest(f"Skipping vLLM colocate test due to memory constraints: {e}")
                    else:
                        raise
        finally:
            # Restore original environment variables
            for key, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value

            release_memory(model, trainer)

    @require_vllm
    def test_training_vllm(self):
        """Test that training works with vLLM for generation."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # increase the learning rate to speed up the test
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
            logging_strategy="no",
            use_vllm=True,
        )

        try:
            trainer = GRPOTrainer(
                model="Qwen/Qwen2.5-0.5B-Instruct",  # tiny models are too small for vLLM
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

        except Exception as e:
            # If vLLM fails to initialize due to hardware constraints or other issues, that's expected
            if any(
                keyword in str(e).lower()
                for keyword in [
                    "outofmemoryerror",
                    "cuda",
                    "memory",
                    "insufficient",
                    "no such device",
                    "free memory",
                    "gpu memory utilization",
                    "decrease gpu memory",
                ]
            ):
                self.skipTest(f"Skipping vLLM training test due to hardware constraints: {e}")
            elif "KeyError" in str(e) and "RANK" in str(e):
                self.skipTest(f"Skipping vLLM training test due to environment setup issues: {e}")
            elif "ValueError" in str(e) and "memory" in str(e).lower():
                self.skipTest(f"Skipping vLLM training test due to memory constraints: {e}")
            else:
                raise

        release_memory(trainer.model, trainer)
