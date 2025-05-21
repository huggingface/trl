"""
Example script for training a GRPO model on PubMed VQA dataset with LoRA.
Toy dataset is from https://huggingface.co/datasets/FreedomIntelligence/Medical_Multimodal_Evaluation_Data/discussions

Usage:
    python grpo_vqa_gemma3.py \
    --model_id google/gemma-3-4b-it \
    --dataset_name bocciDeRock/medical_multimodal_eval_100_examples \
    --output_dir output_dir \
    --num_train_epochs 2
"""

import argparse
import re

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a GRPO model with LoRA on an image-question dataset."
    )
    parser.add_argument(
        "--model_id", type=str, required=True,
        help="Path or identifier of pretrained model"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="bocciDeRock/pubMed100",
        help="HuggingFace dataset name for VQA"
    )
    parser.add_argument(
        "--dataset_split", type=str, default="train",
        help="Split of the dataset to use"
    )
    parser.add_argument(
        "--output_dir", type=str, default="Gemma3-4b-vision-GRPO-test",
        help="Directory to save fine-tuned model"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5,
        help="Learning rate for training"
    )
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=2,
        help="Batch size per device"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=2,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--max_prompt_length", type=int, default=128,
        help="Maximum prompt length"
    )
    parser.add_argument(
        "--max_completion_length", type=int, default=256,
        help="Maximum completion length"
    )
    parser.add_argument(
        "--num_generations", type=int, default=2,
        help="Number of generations per sample for reward evaluation"
    )
    return parser.parse_args()


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the "
    "answer. The reasoning process and answer are enclosed within <think></think> and <answer></answer> tags."
)


def make_conversation(sample):
    images = sample.get("image", [])[:5]
    prompt = f"choose the correct answer from options\nquestion:{sample['question']}\noptions: {str(sample['options'])}"
    prompt_content = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content":
            [{"type": "text", "text": prompt}] +
            [{"type": "image", "image": img} for img in images]
        },
    ]
    return {"prompt": prompt_content, "gold": sample["answer"]}


def format_reward(completions, **kwargs):
    pattern = re.compile(r"^<think>.*?</think>\s*<answer>.*?</answer>$", re.DOTALL)
    rewards = []
    for comp in completions:
        text = comp[0]["content"]
        rewards.append(1.0 if pattern.match(text) else 0.0)
    return rewards


def main():
    args = parse_args()
    raw_ds = load_dataset(args.dataset_name, split=args.dataset_split)
    train_data = [make_conversation(sample) for sample in raw_ds]

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_id)

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        remove_unused_columns=False,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        bf16=False,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        report_to=["none"],
        logging_steps=1,
        save_strategy="steps",
        save_steps=100,
        log_completions=True,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward],
        args=training_args,
        train_dataset=train_data,
        # Use Gemma-3’s processor for image inputs;
        # omit if there are no images (or pass a PreTrainedTokenizerBase instance instead).
        processing_class=processor,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()


