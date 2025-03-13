from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText
from trl.trainer.utils import pad
import torch


if __name__ == "__main__":
    # Load dataset
    train_dataset = load_dataset("open-r1/codeforces-cots", split="train[:100]")
    train_dataset = train_dataset.remove_columns("prompt")

    # Load model and processor
    model_id = "google/gemma-3-4b-it"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id, attn_implementation="eager")

    def pad_gemma(examples, return_tensors="pt", pad_to_multiple_of=None):
        input_ids = [torch.tensor(example["input_ids"]) for example in examples]
        attention_mask = [torch.ones(len(example["input_ids"]), dtype=torch.int) for example in examples]
        input_ids = pad(input_ids, processor.tokenizer.pad_token_id)
        attention_mask = pad(attention_mask, 0)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    # Add a method "pad" to the processor
    processor.pad = pad_gemma
    processor.pad_token_id = processor.tokenizer.pad_token_id

    # Train model
    training_args = SFTConfig(
        output_dir=f"{model_id}-codeforces-SFT",
        logging_steps=10,
        bf16=True,
        use_liger_kernel=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # max_length=16384*2,
        max_length=32,
        # packing=True,
        per_device_train_batch_size=1,
        # dataset_num_proc=32,
    )
    trainer = SFTTrainer(
        args=training_args,
        model=model,
        processing_class=processor,
        train_dataset=train_dataset,
    )
    trainer.train()

    # Push to hub
    trainer.push_to_hub(dataset_name="open-r1/codeforces-cots")
