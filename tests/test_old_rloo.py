import tempfile
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import RLOOConfig, RLOOTrainer


def reward_func(completions, **kwargs):
    """Reward function that rewards completions with more unique letters."""
    return [float(len(set(completion))) for completion in completions]


def main():
    model_id = "Qwen/Qwen3-0.6B"
    policy_model = AutoModelForCausalLM.from_pretrained(model_id)
    policy_ref_model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    with tempfile.TemporaryDirectory() as tmp_dir:
        training_args = RLOOConfig(
            output_dir=tmp_dir,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            total_episodes=1,
            num_train_epochs=1,
            max_steps=2,
            rloo_k=2, 
            report_to="none",
            learning_rate=1e-6,  # Match GRPO default
            temperature=1.0,     # Match GRPO default
        )

        # Create a simple dataset
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        prompts = [tokenizer.apply_chat_template(dataset[i]["prompt"], tokenize=False) for i in range(len(dataset))]
        tokenized = tokenizer(prompts, padding=True, padding_side="right", return_tensors="pt")
        dummy_dataset = Dataset.from_dict({"input_ids": tokenized["input_ids"].tolist()})

        trainer = RLOOTrainer(
            config=training_args,
            policy=policy_model,
            reward_model=reward_func,
            ref_policy=policy_ref_model,
            processing_class=tokenizer,
            train_dataset=dummy_dataset,
            eval_dataset=dummy_dataset,
        )

        # Test that training completes without errors
        trainer.train()

        # Check if objective/rlhf_reward is available
        print("Training completed successfully!")
        print(f"Last log entry: {trainer.state.log_history[-1]}")
        if "objective/rlhf_reward" in trainer.state.log_history[-1]:
            print("✓ objective/rlhf_reward found in logs")
        else:
            print("✗ objective/rlhf_reward not found in logs")


if __name__ == "__main__":
    main()
