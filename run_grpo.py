# train_grpo.py
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import argparse
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
import re

def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            try:
                reward = float(verify(answer_parsed, gold_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = 0.0
        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 1.0
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GRPO model with custom reward function.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Output directory for the trained model.")
    parser.add_argument("--logging_steps", type=int, default=1, help="Number of steps between logging.")
    parser.add_argument("--max_completion_length", type=int, default=4000, help="Maximum length of completions.")
    parser.add_argument("--num_iterations", type=int, default=2, help="Number of training iterations.")
    parser.add_argument("--use_vllm_logprobs", type=lambda x: x.lower() in ['true', '1', 'yes'], default=False, help="Whether to use vLLM log probabilities.")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>", help="System prompt to prepend to the user prompt.")
    
    args = parser.parse_args()

    training_args = GRPOConfig(
        output_dir=f"{args.model}-vllm_logprobs-{args.use_vllm_logprobs}",
        gradient_accumulation_steps=32,
        per_device_train_batch_size=4,        
        logging_steps=args.logging_steps,
        max_completion_length=args.max_completion_length,
        num_generations=14,
        use_vllm=True,
        use_vllm_logprobs=args.use_vllm_logprobs,
        num_iterations=args.num_iterations,
        reward_weights=[1.0, 0.1],
        gradient_checkpointing=True,
        log_completions=True,
        bf16=True,
    )
    
    dataset = load_dataset("open-r1/OpenR1-Math-cn_k12-86k")
    
    def make_conversation(example, prompt_column: str = "problem"):
        prompt = []

        if args.system_prompt is not None:
            prompt.append({"role": "system", "content": args.system_prompt})

        if prompt_column not in example:
            raise ValueError(f"Dataset Question Field Error: {prompt_column} is not supported.")

        prompt.append({"role": "user", "content": example[prompt_column]})
        return {"prompt": prompt}

    dataset = dataset.map(make_conversation)
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")
    reward_funcs = [accuracy_reward, format_reward]
    
    model_kwargs = dict(
        attn_implementation="flash_attention_2",
        torch_dtype="bfloat16",
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs
    
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset["train"],
    )
    trainer.train()