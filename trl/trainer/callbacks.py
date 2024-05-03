from typing import List

from accelerate.utils import gather_object
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    GenerationConfig,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    is_wandb_available,
)

from ..models.utils import unwrap_model_for_generation


if is_wandb_available():
    import wandb


class WinRateCallback(TrainerCallback):
    def __init__(
        self,
        prompts: List[str],
        generation_config: GenerationConfig,
        judge,
        trainer,
        batch_size: int = 4,
    ):
        self.prompts = [
            trainer.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
            for p in prompts
        ]
        self.generation_config = generation_config
        self.completions = []
        self.judge = judge
        self.ref_completions = []
        self.trainer = trainer
        self.batch_size = batch_size

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = self.trainer.model_wrapped
        tokenizer = kwargs["tokenizer"]
        tokenizer.padding_side = "left"
        accelerator = self.trainer.accelerator

        with accelerator.split_between_processes(self.prompts, apply_padding=True) as prompts:
            with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
                unwrapped_model.eval()
                for idx in tqdm(
                    range(0, len(prompts), self.batch_size), desc="Generating reference model completions for win rate"
                ):
                    batch = prompts[idx : idx + self.batch_size]
                    tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(
                        model.device
                    )
                    generations = unwrapped_model.generate(
                        **tokenized_batch,
                        generation_config=self.generation_config,
                    )
                    for prompt, generation in zip(tokenized_batch.input_ids, generations):
                        # Remove prompt from generation
                        generation = generation[len(prompt) :]
                        completion = tokenizer.decode(generation, skip_special_tokens=True)
                        self.ref_completions.append(completion)

                unwrapped_model.train()

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = self.trainer.model_wrapped
        tokenizer = kwargs["tokenizer"]
        accelerator = self.trainer.accelerator

        with accelerator.split_between_processes(self.prompts, apply_padding=True) as prompts:
            annotation_batch = {"prompts": prompts, "completions": []}

            with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
                unwrapped_model.eval()
                for idx in tqdm(range(0, len(prompts), self.batch_size), desc="Generating completions for win rate"):
                    batch = prompts[idx : idx + self.batch_size]
                    tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(
                        model.device
                    )
                    generations = unwrapped_model.generate(
                        **tokenized_batch,
                        generation_config=self.generation_config,
                    )
                    for batch_idx, (prompt, generation) in enumerate(zip(tokenized_batch.input_ids, generations)):
                        # Remove prompt from generation
                        generation = generation[len(prompt) :]
                        response_0 = tokenizer.decode(generation, skip_special_tokens=True)
                        response_1 = self.ref_completions[idx + batch_idx]
                        annotation_batch["completions"].append([response_0, response_1])

                unwrapped_model.train()

            # TODO, rerun with order or responses swapped and average
            results_dict = self.judge.judge_batch(annotation_batch["prompts"], annotation_batch["completions"])
            results_dict = Dataset.from_dict(
                {
                    "results": results_dict,
                    "prompts": annotation_batch["prompts"],
                    "completions": annotation_batch["completions"],
                }
            )  # maybe just map the original dataset for logging
            results_dict = gather_object(results_dict)

        # Logging
        if accelerator.is_main_process:
            dataset_len = len(self.prompts)
            results_dataset = Dataset.from_list(results_dict).select(range(dataset_len))

            win_rate = sum([r == 0 for r in results_dataset["results"]]) / len(results_dataset)
            self.trainer.log({"win_rate": win_rate})

            if is_wandb_available():
                wandb.log({"eval_win_rate": win_rate, "train/global_step": state.global_step})
                prompts = results_dataset["prompts"]
                policy = [c[0] for c in results_dataset["completions"]]
                ref = [c[1] for c in results_dataset["completions"]]
                chosen_indices = results_dataset["results"]
                self.trainer.log(
                    {
                        "winrate_generations": wandb.Table(
                            columns=["Prompt", "Policy", "Ref Model", "Chosen index"],
                            rows=[  # TODO replace with zip unpacking
                                [prompt, pol, ref, index]
                                for prompt, pol, ref, index in zip(prompts, policy, ref, chosen_indices)
                            ],
                        )
                    }
                )
                # pop Table otherwise it is included in the history which cannot be pickled and causes an error
                self.trainer.state.log_history.pop()
