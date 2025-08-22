import tempfile
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.trainer.rloo_final_trainer import RLOOFinalTrainer
from trl.trainer.rloo_finall_config import RLOOConfig_NEW

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
        training_args = RLOOConfig_NEW(
            output_dir=tmp_dir,
            per_device_train_batch_size=2,
            learning_rate=1e-6,
            num_generations=2,
            report_to="none",
            beta=0,
            max_steps=6,
            importance_sampling_level="sequence",
            log_completions=True,
            num_completions_to_print=2,
            logging_steps=1,
            num_iterations=1,
        )
        
        trainer = RLOOFinalTrainer(
            model=policy_model,
            reward_funcs=reward_func,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        # Check initial weights
        initial_weights = policy_model.state_dict()
        print("Sample weights before training:")
        for name, param in list(initial_weights.items())[:3]:
            print(f"{name}: {param.flatten()[:5]}")
        
        trainer.train()
        
        # Check final weights
        final_weights = policy_model.state_dict()
        print("\nSample weights after training:")
        for name, param in list(final_weights.items())[:3]:
            print(f"{name}: {param.flatten()[:5]}")
        
        # Check if weights changed
        weights_changed = False
        for name in initial_weights:
            if not torch.equal(initial_weights[name], final_weights[name]):
                weights_changed = True
                break
        print(f"\nWeights changed: {weights_changed}")
        print("Training completed successfully!")


if __name__ == "__main__":
    main()
