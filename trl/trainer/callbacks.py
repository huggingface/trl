# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Union

import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import is_deepspeed_available
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress
from transformers import PreTrainedModel
from transformers.trainer import TrainerCallback
from transformers.trainer_utils import has_length


if is_deepspeed_available():
    import deepspeed
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

class SyncRefModelCallback(TrainerCallback):
    def __init__(
        self,
        ref_model: Union[PreTrainedModel, torch.nn.Module],
        accelerator: Optional[Accelerator],
    ):
        self.accelerator = accelerator
        self.ref_model = ref_model

    @staticmethod
    def _sync_target_model(model, target_model, alpha):
        for target_param, copy_param in zip(target_model.parameters(), model.parameters()):
            target_param.data.mul_(1.0 - alpha).add_(copy_param.data, alpha=alpha)

    @staticmethod
    def sync_target_model(model, target_model, alpha):
        deepspeed_plugin = AcceleratorState().deepspeed_plugin
        if deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3:
            with deepspeed.zero.GatheredParameters(list(model.parameters()), modifier_rank=0):
                if deepspeed.comm.get_rank() == 0:
                    SyncRefModelCallback._sync_target_model(model, target_model, alpha)
        else:
            SyncRefModelCallback._sync_target_model(model, target_model, alpha)

    def on_step_end(self, args, state, control, **kwargs):
        model: PreTrainedModel = kwargs["model"]

        if self.ref_model is not None and state.global_step % args.ref_model_sync_steps == 0:
            if self.accelerator:
                model = self.accelerator.unwrap_model(model)
            self.sync_target_model(model, self.ref_model, args.ref_model_mixup_alpha)


class RichProgressCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that displays the progress of training or evaluation using Rich.
    """

    def __init__(self):
        self.training_bar = None
        self.prediction_bar = None

        self.training_task_id = None
        self.prediction_task_id = None

        self.rich_group = None
        self.rich_console = None

        self.training_status = None
        self.current_step = None

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.training_bar = Progress()
            self.prediction_bar = Progress()

            self.rich_console = Console()

            self.training_status = self.rich_console.status("Nothing to log yet ...")

            self.rich_group = Live(Panel(Group(self.training_bar, self.prediction_bar, self.training_status)))
            self.rich_group.start()

            self.training_task_id = self.training_bar.add_task("[blue]Training the model", total=state.max_steps)
            self.current_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.training_bar.update(self.training_task_id, advance=state.global_step - self.current_step, update=True)
            self.current_step = state.global_step

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if state.is_world_process_zero and has_length(eval_dataloader):
            if self.prediction_task_id is None:
                self.prediction_task_id = self.prediction_bar.add_task(
                    "[blue]Predicting on the evaluation dataset", total=len(eval_dataloader)
                )
            self.prediction_bar.update(self.prediction_task_id, advance=1, update=True)

    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            if self.prediction_task_id is not None:
                self.prediction_bar.remove_task(self.prediction_task_id)
                self.prediction_task_id = None

    def on_predict(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            if self.prediction_task_id is not None:
                self.prediction_bar.remove_task(self.prediction_task_id)
                self.prediction_task_id = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and self.training_bar is not None:
            _ = logs.pop("total_flos", None)
            self.training_status.update(f"[bold green]Status = {str(logs)}")

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.rich_group.stop()

            self.training_bar = None
            self.prediction_bar = None
            self.training_task_id = None
            self.prediction_task_id = None
            self.rich_group = None
            self.rich_console = None
            self.training_status = None
            self.current_step = None


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
