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
from typing import List, Optional, Union

import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import gather_object, is_deepspeed_available
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress
from transformers import (
    GenerationConfig,
    PreTrainedModel,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.integrations import WandbCallback
from transformers.trainer_utils import has_length

from ..models.utils import unwrap_model_for_generation
from .judges import BasePairwiseJudge
from .utils import decode_and_strip_padding, truncate_right


if is_deepspeed_available():
    import deepspeed


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
            with deepspeed.zero.GatheredParameters(
                list(model.parameters()) + list(target_model.parameters()), modifier_rank=0
            ):
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
    """
    A [`~transformers.TrainerCallback`] that computes the win rate of a model based on a reference.

    It generates completions using prompts from the evaluation dataset and compares the trained model's outputs against
    a reference. The reference is either the initial version of the model (before training) or the reference model, if
    available in the trainer. During each evaluation step, a judge determines how often the trained model's completions
    win against the reference using a judge. The win rate is then logged in the trainer's logs under the key
    `"eval_win_rate"`.

    Usage:
    ```python
    trainer = DPOTrainer(...)
    win_rate_callback = WinRateCallback(..., trainer=trainer)
    trainer.add_callback(win_rate_callback)
    ```

    Args:
        judge (`BasePairwiseJudge`):
            The judge to use for comparing completions.
        trainer (`Trainer`):
            Trainer to which the callback will be attached. The trainer's evaluation dataset must include a `"prompt"`
            column containing the prompts for generating completions. If the `Trainer` has a reference model (via the
            `ref_model` attribute), it will use this reference model for generating the reference completions;
            otherwise, it defaults to using the initial model.
        generation_config (`GenerationConfig`, *optional*):
            The generation config to use for generating completions.
        batch_size (`int`, *optional*):
            The batch size to use for generating completions. Defaults to 4.
    """

    def __init__(
        self,
        judge: BasePairwiseJudge,
        trainer: Trainer,
        generation_config: Optional[GenerationConfig] = None,
        batch_size: int = 4,
    ):
        self.generation_config = generation_config
        self.judge = judge
        self.ref_completions = []
        self.trainer = trainer
        self.eval_dataset = self.trainer.eval_dataset
        self.batch_size = batch_size

    def generate_completions_for_model(self, model, tokenizer, prompts):
        completions = []
        with unwrap_model_for_generation(model, self.trainer.accelerator) as unwrapped_model:
            unwrapped_model.eval()
            for idx in range(0, len(prompts), self.batch_size):
                batch = prompts[idx : idx + self.batch_size]
                tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
                generations = unwrapped_model.generate(
                    **tokenized_batch,
                    generation_config=self.generation_config,
                )
                for prompt, generation in zip(tokenized_batch.input_ids, generations):
                    # Remove prompt from generation
                    generation = generation[len(prompt) :]
                    completion = tokenizer.decode(generation, skip_special_tokens=True)
                    completions.append(completion)

            unwrapped_model.train()
        return completions

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # When the trainer is initialized, we generate completions for the reference model.
        tokenizer = kwargs["tokenizer"]
        tokenizer.padding_side = "left"
        accelerator = self.trainer.accelerator
        model = getattr(self.trainer, "ref_model", kwargs["model"])  # get the ref model if any, else use the model
        with accelerator.split_between_processes(self.eval_dataset["prompt"], apply_padding=True) as prompts:
            self.ref_completions = self.generate_completions_for_model(model, tokenizer, prompts)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # At every evaluation step, we generate completions for the model and compare them with the reference
        # completions that have been generated at the beginning of training. We then compute the win rate and log it to
        # the trainer.
        model = kwargs["model"]
        tokenizer = kwargs["tokenizer"]
        accelerator = self.trainer.accelerator
        with accelerator.split_between_processes(self.eval_dataset["prompt"], apply_padding=True) as prompts:
            completions = self.generate_completions_for_model(model, tokenizer, prompts)
            completions = list(zip(self.ref_completions, completions))
            winner_indices = self.judge.judge(self.eval_dataset["prompt"], completions)
            winner_indices = gather_object(winner_indices)

        # Logging
        if self.trainer.accelerator.is_main_process:
            win_rate = sum(winner_idx == 1 for winner_idx in winner_indices) / len(winner_indices)
            self.trainer.log({"eval_win_rate": win_rate})


class LogCompletionsCallback(WandbCallback):
    r"""
    A [`~transformers.TrainerCallback`] that logs completions to Weights & Biases.

    Usage:
    ```python
    prompts = ["The capital of France is", "The opposite of up is"]
    trainer = DPOTrainer(..., callbacks=[LogCompletionsCallback(prompts)])
    ```

    Args:
        prompts (`List[str]`):
            The prompts to generate completions for.
        freq (`Optional[int]`, *optional*, defaults to `None`):
            The frequency at which to log completions. If not provided, defaults to `logging_steps`.
    """

    def __init__(self, prompts: List[str], freq: int = None):
        super().__init__()
        self.prompts = prompts
        self.inputs = None  # will be tokenized in on_train_begin
        self.table = []
        self._last_logged_step = -1
        self.freq = freq

    def on_train_begin(self, args, state, control, **kwargs):
        tokenizer = kwargs["tokenizer"]
        self.inputs = tokenizer(self.prompts, return_tensors="pt", padding=True, truncation=True)

    def on_step_end(self, args, state, control, **kwargs):
        # Only log from the main process
        if not state.is_world_process_zero:
            return

        # Only log once per step (this method may be called multiple times)
        if state.global_step == self._last_logged_step:
            return

        # Only log every `freq` steps (if no `freq` is provided, log every `logging_steps` steps)
        freq = self.freq or state.logging_steps
        if state.global_step % freq != 0:
            return

        # Get the model and tokenizer
        model = kwargs["model"]
        tokenizer = kwargs["tokenizer"]
        model.eval()

        # Generate completions
        generation_config = GenerationConfig(max_new_tokens=args.max_new_tokens, min_new_tokens=args.max_new_tokens)
        inputs = self.inputs.to(args.device)
        _, context_length = inputs["input_ids"].shape
        output = model.generate(**inputs, generation_config=generation_config)

        # Get only the completions
        completion_ids = output[:, context_length:]

        # After the first EOS token, replace all tokens with padding tokens
        completion_ids, _ = truncate_right(completion_ids, tokenizer.eos_token_id, tokenizer.pad_token_id)

        # Decode the prompts and completions
        prompts = decode_and_strip_padding(inputs["input_ids"], tokenizer)
        completions = decode_and_strip_padding(completion_ids, tokenizer)

        # Build the data to log
        global_step = [str(state.global_step)] * len(prompts)
        data = list(zip(global_step, prompts, completions))
        self.table.extend(data)
        table = self._wandb.Table(columns=["step", "prompt", "completion"], data=self.table)
        self._wandb.log({"completions": table})

        # Save the last logged step, so we don't log the same completions multiple times
        self._last_logged_step = state.global_step
