import tempfile
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


def reward_func(completions, **kwargs):
    """Reward function that rewards completions with more unique letters."""
    completion_contents = [completion[0]["content"] for completion in completions]
    return [float(len(set(content))) for content in completion_contents]


def main():
    model_id = "Qwen/Qwen3-0.6B"
    policy_model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")


    with tempfile.TemporaryDirectory() as tmp_dir:
        training_args = GRPOConfig(
            output_dir=tmp_dir,
            per_device_train_batch_size=2,
            learning_rate=0.1,
            num_generations=2,
            report_to="none",
            beta=0.05,
            max_steps=2,
            loss_type="grpo",
            importance_sampling_level="sequence",
        )
        
        trainer = GRPOTrainer(
            model=policy_model,
            reward_funcs=reward_func,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        trainer.train()
        print("Training completed successfully!")


if __name__ == "__main__":
    main()
