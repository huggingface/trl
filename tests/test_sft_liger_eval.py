from datasets import Dataset
import pytest
import torch
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

@pytest.mark.slow
def test_liger_eval_vram_and_token_accuracy():
    # Setup dummy model and tokenizer
    model_name = "hf-internal-testing/tiny-random-GPT2LMHeadModel"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Dummy input
    input_text = "Hello world"
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs["labels"] = inputs["input_ids"].clone()


    # Use SFTConfig for args to avoid missing attributes
    args = SFTConfig(
        output_dir="/tmp",
        use_liger_kernel=True,
        prediction_loss_only=False,
        loss_type="dft",
        bf16=False,
    )

    # Patch SFTTrainer to inject args and accelerator
    class DummyAccelerator:
        def gather_for_metrics(self, x):
            return x


    # Minimal dummy train_dataset using HuggingFace Dataset
    train_dataset = Dataset.from_dict({
        "input_ids": inputs["input_ids"],
        "labels": inputs["labels"]
    })

    trainer = SFTTrainer(model=model, args=args, train_dataset=train_dataset, processing_class=tokenizer)
    trainer.accelerator = DummyAccelerator()
    trainer.compute_metrics = None
    trainer.model.training = False

    # Should not OOM or crash on token_accuracy
    try:
        trainer.compute_loss(model, inputs)
    except Exception as e:
        pytest.fail(f"SFTTrainer.compute_loss crashed: {e}")
