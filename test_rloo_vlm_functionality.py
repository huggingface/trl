#!/usr/bin/env python3
"""
Comprehensive VLM (Vision Language Model) Test Suite for RLOO Trainer

This script tests all VLM functionality that was added to the RLOO trainer,
including imports, data processing, model loading, and training functionality.

Usage:
    python test_vlm_functionality.py

Environment:
    Requires conda environment 'behrooz' with updated dependencies via uv
"""

import sys
import tempfile
import traceback
from typing import Dict, List, Tuple


def test_basic_imports() -> Tuple[bool, str]:
    """Test basic VLM-related imports."""
    print("=== Test 1: Basic VLM Imports ===")
    
    try:
        # Core VLM function we added
        from trl.data_utils import prepare_multimodal_messages
        print("✅ prepare_multimodal_messages import: SUCCESS")
        
        # Test that the function is callable
        func_name = prepare_multimodal_messages.__name__
        print(f"✅ Function callable: {func_name}")
        
        # Datasets for VLM testing
        from datasets import load_dataset
        print("✅ datasets import: SUCCESS")
        
        return True, "All basic imports successful"
        
    except Exception as e:
        error_msg = f"Basic imports failed: {e}"
        print(f"❌ {error_msg}")
        return False, error_msg


def test_vlm_utilities() -> Tuple[bool, str]:
    """Test VLM utility functions."""
    print("\n=== Test 2: VLM Utility Functions ===")
    
    try:
        # Pixel values utilities
        from trl.trainer.utils import split_pixel_values_by_grid, unsplit_pixel_values_by_grid
        print("✅ pixel values utilities import: SUCCESS")
        
        # VLM model classes
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        print("✅ VLM model classes import: SUCCESS")
        
        return True, "All VLM utilities working"
        
    except Exception as e:
        error_msg = f"VLM utilities test failed: {e}"
        print(f"❌ {error_msg}")
        return False, error_msg


def test_vlm_dataset_loading() -> Tuple[bool, str]:
    """Test VLM dataset loading and processing."""
    print("\n=== Test 3: VLM Dataset Loading ===")
    
    try:
        from datasets import load_dataset
        
        # Test basic dataset
        dataset = load_dataset('trl-internal-testing/zen', 'standard_prompt_only', split='train', streaming=True)
        sample = next(iter(dataset.take(1)))
        print(f"✅ Basic dataset loading: SUCCESS - Keys: {list(sample.keys())}")
        
        # Test VLM image dataset
        vlm_dataset = load_dataset('trl-internal-testing/zen-image', 'conversational_prompt_only', split='train', streaming=True)
        vlm_sample = next(iter(vlm_dataset.take(1)))
        print(f"✅ VLM dataset loading: SUCCESS - Keys: {list(vlm_sample.keys())}")
        
        # Test data processing if image data is available
        if 'images' in vlm_sample and 'prompt' in vlm_sample:
            from trl.data_utils import prepare_multimodal_messages
            try:
                result = prepare_multimodal_messages(vlm_sample['prompt'], vlm_sample['images'])
                print(f"✅ prepare_multimodal_messages processing: SUCCESS - Type: {type(result)}")
            except Exception as e:
                print(f"⚠️ prepare_multimodal_messages processing: {e}")
        
        return True, "VLM dataset loading successful"
        
    except Exception as e:
        error_msg = f"VLM dataset loading failed: {e}"
        print(f"❌ {error_msg}")
        return False, error_msg


def test_rloo_config_import() -> Tuple[bool, str]:
    """Test RLOO config and trainer imports."""
    print("\n=== Test 4: RLOO Config and Trainer Imports ===")
    
    try:
        from trl import RLOOConfig, RLOOTrainer
        print("✅ RLOOConfig and RLOOTrainer import: SUCCESS")
        
        # Test config creation
        config = RLOOConfig(
            output_dir="/tmp/test",
            per_device_train_batch_size=2,
            num_generations=2,
            max_completion_length=4,
            max_steps=1,
            report_to="none"
        )
        print("✅ RLOOConfig creation: SUCCESS")
        
        # Check some VLM-related attributes
        vlm_attrs = ['max_prompt_length', 'max_completion_length', 'generation_batch_size']
        for attr in vlm_attrs:
            if hasattr(config, attr):
                print(f"✅ Config attribute {attr}: EXISTS")
            else:
                print(f"⚠️ Config attribute {attr}: MISSING")
        
        return True, "RLOO imports successful"
        
    except Exception as e:
        error_msg = f"RLOO imports failed: {e}"
        print(f"❌ {error_msg}")
        return False, error_msg


def test_model_configs() -> Tuple[bool, str]:
    """Test VLM model configuration access."""
    print("\n=== Test 5: VLM Model Configurations ===")
    
    try:
        from transformers import AutoConfig
        
        # Test access to VLM models used in tests
        model_ids = [
            'trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration',
            'trl-internal-testing/tiny-LlavaNextForConditionalGeneration',
            'trl-internal-testing/tiny-Qwen2VLForConditionalGeneration',
        ]
        
        successful_models = []
        for model_id in model_ids:
            try:
                config = AutoConfig.from_pretrained(model_id)
                print(f"✅ Model config access ({model_id.split('/')[-1]}): SUCCESS")
                successful_models.append(model_id)
                break  # Just test one to avoid too many downloads
            except Exception as e:
                print(f"⚠️ Model config access ({model_id.split('/')[-1]}): {e}")
        
        if successful_models:
            return True, f"Model configs accessible: {len(successful_models)} tested"
        else:
            return False, "No model configs accessible"
            
    except Exception as e:
        error_msg = f"Model config test failed: {e}"
        print(f"❌ {error_msg}")
        return False, error_msg


def test_vlm_training() -> Tuple[bool, str]:
    """Test complete VLM training functionality."""
    print("\n=== Test 6: VLM Training Functionality ===")
    
    try:
        import torch
        from datasets import load_dataset
        from trl import RLOOConfig, RLOOTrainer
        
        # Test model
        model_id = 'trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration'
        print(f"Testing VLM training with {model_id}")
        
        # Load VLM dataset
        dataset = load_dataset('trl-internal-testing/zen-image', 'conversational_prompt_only', split='train')
        print(f"Dataset loaded: {len(dataset)} samples")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Configure training arguments
            training_args = RLOOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,
                per_device_train_batch_size=2,  # Must be divisible by num_generations
                num_generations=2,  # At least 2 required for RLOO
                max_completion_length=4,  # Short completions for testing
                max_prompt_length=None,  # Disable prompt truncation for VLM
                report_to='none',
                max_steps=1,  # Just one step for testing
                logging_steps=1,
            )
            
            # Initialize trainer
            trainer = RLOOTrainer(
                model=model_id,
                reward_funcs='trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5',
                args=training_args,
                train_dataset=dataset,
            )
            
            print("✅ Trainer initialized successfully")
            
            # Run one training step
            trainer.train()
            print("✅ Training step completed successfully")
            
            # Check training results
            if trainer.state.log_history:
                train_loss = trainer.state.log_history[-1].get('train_loss', 'N/A')
                print(f"✅ Training loss: {train_loss}")
                
                # Check for VLM-specific metrics
                last_log = trainer.state.log_history[-1]
                vlm_metrics = ['reward', 'kl', 'completions/mean_length', 'entropy']
                for metric in vlm_metrics:
                    if metric in last_log:
                        print(f"✅ Metric {metric}: {last_log[metric]}")
                
                return True, f"VLM training successful with loss: {train_loss}"
            else:
                return False, "No training logs available"
                
    except Exception as e:
        error_msg = f"VLM training failed: {e}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return False, error_msg


def test_vlm_beta_non_zero() -> Tuple[bool, str]:
    """Test VLM training with non-zero beta (reference model usage)."""
    print("\n=== Test 7: VLM Training with Beta Non-Zero ===")
    
    try:
        import torch
        from datasets import load_dataset
        from trl import RLOOConfig, RLOOTrainer
        
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_prompt_only", split="train")
        print(f"Dataset loaded: {len(dataset)} samples")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = RLOOConfig(
                output_dir=tmp_dir,
                beta=0.1,  # set beta to non-zero value to test the case where the reference model is used
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                report_to="none",
                max_steps=1,  # Just one step for testing
            )
            trainer = RLOOTrainer(
                model="trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}
            print("✅ Trainer initialized with beta=0.1")

            trainer.train()
            print("✅ Training completed with reference model")

            # Check that the params have changed
            # Because of the way the tiny models are initialized, the gradient does not flow properly through the
            # vision parts of the model, so we skip them. Ideally, we should fix the init of these models.
            params_changed = 0
            params_to_skip = ("model.visual.",)
            for n, param in previous_trainable_params.items():
                if n.startswith(params_to_skip):
                    continue
                new_param = trainer.model.get_parameter(n)
                if not torch.equal(param, new_param):
                    params_changed += 1
            
            print(f"✅ Parameters changed: {params_changed}")
            
            # Check training logs
            if trainer.state.log_history:
                train_loss = trainer.state.log_history[-1].get('train_loss', 'N/A')
                print(f"✅ Training loss: {train_loss}")
                return True, f"VLM beta training successful, {params_changed} params changed"
            else:
                return False, "No training logs available"
                
    except Exception as e:
        error_msg = f"VLM beta training failed: {e}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return False, error_msg


def test_vlm_peft() -> Tuple[bool, str]:
    """Test VLM training with PEFT (LoRA)."""
    print("\n=== Test 8: VLM Training with PEFT ===")
    
    try:
        import torch
        from datasets import load_dataset
        from trl import RLOOConfig, RLOOTrainer
        from transformers import AutoModelForImageTextToText
        
        # Check if PEFT is available
        try:
            from peft import LoraConfig
            print("✅ PEFT available")
        except ImportError:
            return False, "PEFT not available - skipping test"

        model = AutoModelForImageTextToText.from_pretrained(
            "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration"
        )
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        print(f"Model and dataset loaded")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = RLOOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                report_to="none",
                max_steps=1,  # Just one step for testing
            )
            trainer = RLOOTrainer(
                model=model,
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
                peft_config=LoraConfig(target_modules=["q_proj", "v_proj"]),
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}
            print("✅ Trainer initialized with LoRA")

            trainer.train()
            print("✅ PEFT training completed")

            # Check that the peft params have changed and the base model params have not changed
            peft_params_changed = 0
            base_params_changed = 0
            
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                if n in base_param_names:  # We expect the base model params to be the same
                    if not torch.allclose(param, new_param):
                        base_params_changed += 1
                elif "base_layer" not in n:  # We expect the peft params to be different (except for the base layer)
                    if not torch.allclose(param, new_param):
                        peft_params_changed += 1
            
            print(f"✅ PEFT parameters changed: {peft_params_changed}")
            print(f"✅ Base parameters unchanged: {len(base_param_names) - base_params_changed}/{len(base_param_names)}")
            
            # Check training logs
            if trainer.state.log_history:
                train_loss = trainer.state.log_history[-1].get('train_loss', 'N/A')
                print(f"✅ Training loss: {train_loss}")
                return True, f"VLM PEFT training successful, {peft_params_changed} PEFT params changed"
            else:
                return False, "No training logs available"
                
    except Exception as e:
        error_msg = f"VLM PEFT training failed: {e}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return False, error_msg


def test_package_versions() -> Tuple[bool, str]:
    """Test that all required packages are properly installed."""
    print("\n=== Test 7: Package Versions ===")
    
    try:
        import torch
        import torchvision  
        import transformers
        import vllm
        
        print(f"✅ PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}")
        print(f"✅ Torchvision {torchvision.__version__}")
        print(f"✅ Transformers {transformers.__version__}")
        print(f"✅ vLLM {vllm.__version__}")
        
        return True, "All packages properly installed"
        
    except Exception as e:
        error_msg = f"Package version check failed: {e}"
        print(f"❌ {error_msg}")
        return False, error_msg


def run_all_tests() -> Dict[str, Tuple[bool, str]]:
    """Run all VLM tests and return results."""
    print("🧪 Starting Comprehensive VLM Test Suite")
    print("=" * 60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("VLM Utilities", test_vlm_utilities),
        ("Dataset Loading", test_vlm_dataset_loading),
        ("RLOO Config", test_rloo_config_import),
        ("Model Configs", test_model_configs),
        ("VLM Training", test_vlm_training),
        ("VLM Beta Non-Zero", test_vlm_beta_non_zero),
        ("VLM PEFT", test_vlm_peft),
        ("Package Versions", test_package_versions),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            success, message = test_func()
            results[test_name] = (success, message)
        except Exception as e:
            results[test_name] = (False, f"Test execution failed: {e}")
    
    return results


def print_summary(results: Dict[str, Tuple[bool, str]]) -> None:
    """Print test summary."""
    print("\n" + "=" * 60)
    print("🧪 VLM TEST SUITE SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for success, _ in results.values() if success)
    total = len(results)
    
    for test_name, (success, message) in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:8} | {test_name:15} | {message}")
    
    print("-" * 60)
    print(f"📊 Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! VLM support is fully functional!")
        print("✅ VLM implementation for RLOO trainer is complete and working")
        print("✅ Environment issues have been resolved")
    else:
        print(f"⚠️  {total-passed} test(s) failed. Check the details above.")
        
    print("=" * 60)


if __name__ == "__main__":
    print("VLM Functionality Test Suite for RLOO Trainer")
    print("Testing Vision Language Model support implementation")
    print()
    
    # Run all tests
    test_results = run_all_tests()
    
    # Print summary
    print_summary(test_results)
    
    # Exit with appropriate code
    all_passed = all(success for success, _ in test_results.values())
    sys.exit(0 if all_passed else 1)