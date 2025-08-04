import tempfile
import unittest
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from trl import RLOOConfig, RLOOTrainer


class RLOOTrainerTester(unittest.TestCase):
    def setUp(self):
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.policy_model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        self.policy_ref_model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side="right")
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def test_rloo_training_with_custom_reward(self):
        def reward_func(completions, **kwargs):
            """Reward function that rewards completions with more unique letters."""
            return [float(len(set(completion))) for completion in completions]

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = RLOOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                total_episodes=1,
                num_train_epochs=1,
                max_steps=2,
                rloo_k=2,
                learning_rate=0.1,  
                report_to="none",
            )

            # Create a simple dataset
            dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
            prompts = [self.tokenizer.apply_chat_template(dataset[i]["prompt"], tokenize=False) for i in range(len(dataset))]
            tokenized = self.tokenizer(prompts, padding=True, padding_side="right", return_tensors="pt")
            dummy_dataset = Dataset.from_dict({"input_ids": tokenized["input_ids"].tolist()})

            trainer = RLOOTrainer(
                config=training_args,
                policy=self.policy_model,
                reward_model=reward_func,
                ref_policy=self.policy_ref_model,
                processing_class=self.tokenizer,
                train_dataset=dummy_dataset,
                eval_dataset=dummy_dataset,
            )

            # Test that training completes without errors
            trainer.train()

            # Check if objective/rlhf_reward is available
            self.assertIn("objective/rlhf_reward", trainer.state.log_history[-1])
