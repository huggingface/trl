import unittest
import gc
import torch
from datasets import load_dataset
from parameterized import parameterized
from transformers.testing_utils import require_torch_accelerator
import itertools
from .testing_constants import MODELS_TO_TEST, PACKING_OPTIONS
from trl import GRPOConfig, GRPOTrainer
from accelerate.utils.memory import release_memory
import tempfile
from transformers.testing_utils import require_liger_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer

@require_torch_accelerator
class GRPOTrainerSlowTester(unittest.TestCase):
    def setUp(self):
        self.train_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        self.eval_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="test")
        self.max_length = 128

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    @parameterized.expand(MODELS_TO_TEST)
    @require_liger_kernel
    def test_training_with_liger_grpo_loss(self, model_name):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=3,
                num_generations=3,
                use_liger_grpo_loss=True,
                max_completion_length=self.max_length,
                max_steps=4,
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
            trainer.train()

        release_memory(trainer.model, trainer)
