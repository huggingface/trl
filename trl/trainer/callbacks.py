# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import os
from typing import List, Optional, Union

import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import gather_object, is_comet_ml_available, is_deepspeed_available, is_wandb_available
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress
from transformers import (
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_utils import has_length

from ..data_utils import maybe_apply_chat_template
from ..import_utils import is_mergekit_available
from ..mergekit_utils import MergeConfig, merge_models, upload_model_to_hf
from ..models.utils import unwrap_model_for_generation
from .judges import BasePairwiseJudge
from .utils import log_table_to_comet_experiment


if is_deepspeed_available():
    import deepspeed

if is_comet_ml_available():
    pass

if is_wandb_available():
    import wandb


def _generate_completions(
    prompts: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    accelerator: Accelerator,
    generation_config: Optional[GenerationConfig],
    batch_size: int = 1,
) -> list[str]:
    """
    Generates completions for a list of pre-formatted prompts from the given model.

    Args:
        prompts (list[str]): A list of input prompts for which completions are to be generated.
        model (PreTrainedModel): The pre-trained model to be used for generation.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to be used for encoding and decoding.
        accelerator (Accelerator): The accelerator to be used for model execution.
        generation_config (GenerationConfig): Configuration for text generation.
        batch_size (int, optional): The number of prompts to process in each batch. Default is 1.

    Returns:
        list[str]: A list of generated text completions corresponding to the input prompts.
    """
    completions = []
    with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
        for idx in range(0, len(prompts), batch_size):
            batch = prompts[idx : idx + batch_size]
            tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
            generations = unwrapped_model.generate(
                **tokenized_batch,
                generation_config=generation_config,
            )
            for prompt, generation in zip(tokenized_batch.input_ids, generations):
                # Remove prompt from generation
                generation = generation[len(prompt) :]
                completion = tokenizer.decode(generation, skip_special_tokens=True)
                completions.append(completion)
    return completions


class SyncRefModelCallback(TrainerCallback):
    """
    Callback to synchronize the model with a reference model.
    """

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


def _win_rate_completions_df(
    state: TrainerState, prompts: List[str], completions: List[str], winner_indices: List[str]
) -> pd.DataFrame:
    global_step = [str(state.global_step)] * len(prompts)
    data = list(zip(global_step, prompts, completions, winner_indices))
    # Split completions from reference model and policy
    split_data = [(item[0], item[1], item[2][0], item[2][1], item[3]) for item in data]
    return pd.DataFrame(split_data, columns=["step", "prompt", "reference_model", "policy", "winner_index"])


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
    judge = PairRMJudge()
    win_rate_callback = WinRateCallback(judge=judge, trainer=trainer)
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
        num_prompts (`int` or `None`, *optional*, defaults to `None`):
            The number of prompts to generate completions for. If not provided, defaults to the number of examples
            in the evaluation dataset.
        shuffle_order (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the order of the completions before judging.
        use_soft_judge (`bool`, *optional*, defaults to `False`):
            Whether to use a soft judge that returns a win probability between 0 and 1 for the first completion vs the
            second.
    """

    def __init__(
        self,
        judge: BasePairwiseJudge,
        trainer: Trainer,
        generation_config: Optional[GenerationConfig] = None,
        num_prompts: Optional[int] = None,
        shuffle_order: bool = True,
        use_soft_judge: bool = False,
    ):
        self.judge = judge
        self.trainer = trainer
        self.shuffle_order = shuffle_order
        self.generation_config = generation_config
        self.ref_completions = []
        self.use_soft_judge = use_soft_judge

        if self.trainer.eval_dataset is None:
            raise ValueError("Trainer must have an evaluation dataset to use the WinRateCallback.")
        else:
            self.eval_dataset = self.trainer.eval_dataset

        if num_prompts is not None:
            self.eval_dataset = self.eval_dataset.select(range(num_prompts))

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # When the trainer is initialized, we generate completions for the reference model.
        tokenizer = kwargs["processing_class"]
        tokenizer.padding_side = "left"
        accelerator = self.trainer.accelerator
        # Use the reference model if available, otherwise use the initial model
        model = getattr(self.trainer, "ref_model", None)
        # At this point, there are two cases where `ref_model` is None:
        # 1. The method doesn't require a reference model.
        # 2. The method uses a reference model, but `ref_model` is set to None.
        #    This occurs when using PEFT, where the reference model can be obtained by simply disabling the model's adapter.
        #    In theory, we should disable the adapter here, but since it's zero-initialized at the start of training,
        #    the model behaves identically with or without the adapter.
        #    Therefore, there's no need to explicitly disable it at this point.
        if model is None:
            model = self.trainer.model_wrapped
        with accelerator.split_between_processes(self.eval_dataset["prompt"]) as prompts:
            self.ref_completions = _generate_completions(
                prompts,
                model=model,
                tokenizer=tokenizer,
                accelerator=accelerator,
                generation_config=self.generation_config,
                batch_size=args.per_device_eval_batch_size,
            )
            # Compute initial win rate as a reference point
            completions = list(zip(self.ref_completions, self.ref_completions))
            if self.use_soft_judge:
                ref_win_probs = self.judge.judge(prompts, completions, self.shuffle_order, return_scores=True)
                winner_indices = [0 if score > 0.5 else 1 for score in ref_win_probs]
                ref_win_probs = gather_object(ref_win_probs)
            else:
                winner_indices = self.judge.judge(prompts, completions, self.shuffle_order)
            prompts = gather_object(prompts)
            completions = gather_object(completions)
            winner_indices = gather_object(winner_indices)

        # Logging
        if self.trainer.accelerator.is_main_process:
            win_rate = sum(winner_idx == 1 for winner_idx in winner_indices) / len(winner_indices)
            if self.use_soft_judge:
                avg_win_prob = 1.0 - sum(ref_win_probs) / len(ref_win_probs)
                self.trainer.log({"eval_avg_win_prob": avg_win_prob, "eval_win_rate": win_rate})
            else:
                self.trainer.log({"eval_win_rate": win_rate})

            if "wandb" in args.report_to:
                import wandb

                if wandb.run is not None:
                    df = _win_rate_completions_df(
                        state=state,
                        prompts=prompts,
                        completions=completions,
                        winner_indices=winner_indices,
                    )
                    wandb.log({"win_rate_completions": wandb.Table(dataframe=df)})

            if "comet_ml" in args.report_to:
                df = _win_rate_completions_df(
                    state=state,
                    prompts=prompts,
                    completions=completions,
                    winner_indices=winner_indices,
                )
                log_table_to_comet_experiment(
                    name="win_rate_completions.csv",
                    table=df,
                )

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # At every evaluation step, we generate completions for the model and compare them with the reference
        # completions that have been generated at the beginning of training. We then compute the win rate and log it to
        # the trainer.
        tokenizer = kwargs["processing_class"]
        tokenizer.padding_side = "left"
        accelerator = self.trainer.accelerator
        model = self.trainer.model_wrapped
        with accelerator.split_between_processes(self.eval_dataset["prompt"]) as prompts:
            completions = _generate_completions(
                prompts,
                model=model,
                tokenizer=tokenizer,
                accelerator=accelerator,
                generation_config=self.generation_config,
                batch_size=args.per_device_eval_batch_size,
            )

            completions = list(zip(self.ref_completions, completions))

            if self.use_soft_judge:
                ref_win_probs = self.judge.judge(prompts, completions, self.shuffle_order, return_scores=True)
                winner_indices = [0 if score > 0.5 else 1 for score in ref_win_probs]
                ref_win_probs = gather_object(ref_win_probs)
            else:
                winner_indices = self.judge.judge(prompts, completions, self.shuffle_order)
            prompts = gather_object(prompts)
            completions = gather_object(completions)
            winner_indices = gather_object(winner_indices)

        # Logging
        if self.trainer.accelerator.is_main_process:
            win_rate = sum(winner_idx == 1 for winner_idx in winner_indices) / len(winner_indices)
            if self.use_soft_judge:
                avg_win_prob = 1.0 - sum(ref_win_probs) / len(ref_win_probs)
                self.trainer.log({"eval_avg_win_prob": avg_win_prob, "eval_win_rate": win_rate})
            else:
                self.trainer.log({"eval_win_rate": win_rate})

            if "wandb" in args.report_to:
                import wandb

                if wandb.run is not None:
                    df = _win_rate_completions_df(
                        state=state,
                        prompts=prompts,
                        completions=completions,
                        winner_indices=winner_indices,
                    )
                    wandb.log({"win_rate_completions": wandb.Table(dataframe=df)})

            if "comet_ml" in args.report_to:
                df = _win_rate_completions_df(
                    state=state,
                    prompts=prompts,
                    completions=completions,
                    winner_indices=winner_indices,
                )
                log_table_to_comet_experiment(
                    name="win_rate_completions.csv",
                    table=df,
                )


class LogCompletionsCallback(TrainerCallback):
    r"""
    A [`~transformers.TrainerCallback`] that logs completions to Weights & Biases and/or Comet.

    Usage:
    ```python
    trainer = DPOTrainer(...)
    completions_callback = LogCompletionsCallback(trainer=trainer)
    trainer.add_callback(completions_callback)
    ```

    Args:
        trainer (`Trainer`):
            Trainer to which the callback will be attached. The trainer's evaluation dataset must include a `"prompt"`
            column containing the prompts for generating completions.
        generation_config (`GenerationConfig`, *optional*):
            The generation config to use for generating completions.
        num_prompts (`int` or `None`, *optional*):
            The number of prompts to generate completions for. If not provided, defaults to the number of examples in the evaluation dataset.
        freq (`int` or `None`, *optional*):
            The frequency at which to log completions. If not provided, defaults to the trainer's `eval_steps`.
    """

    def __init__(
        self,
        trainer: Trainer,
        generation_config: Optional[GenerationConfig] = None,
        num_prompts: Optional[int] = None,
        freq: Optional[int] = None,
    ):
        self.trainer = trainer
        self.generation_config = generation_config
        self.freq = freq
        self.table = []
        self._last_logged_step = -1

        if self.trainer.eval_dataset is None:
            raise ValueError("Trainer must have an evaluation dataset to use the LogCompletionsCallback.")
        else:
            self.eval_dataset = self.trainer.eval_dataset

        if num_prompts is not None:
            self.eval_dataset = self.eval_dataset.select(range(num_prompts))

    def on_step_end(self, args, state, control, **kwargs):
        # Only log once per step (this method may be called multiple times)
        if state.global_step == self._last_logged_step:
            return

        # Only log every `freq` steps (if no `freq` is provided, log every `eval_steps` steps)
        freq = self.freq or state.eval_steps
        if state.global_step % freq != 0:
            return

        tokenizer = kwargs["processing_class"]
        tokenizer.padding_side = "left"
        accelerator = self.trainer.accelerator
        model = self.trainer.model_wrapped
        with accelerator.split_between_processes(self.eval_dataset["prompt"]) as prompts:
            prompts = [maybe_apply_chat_template({"prompt": prompt}, tokenizer)["prompt"] for prompt in prompts]
            completions = _generate_completions(
                prompts,
                model=model,
                tokenizer=tokenizer,
                accelerator=accelerator,
                generation_config=self.generation_config,
                batch_size=args.per_device_eval_batch_size,
            )
            completions = gather_object(completions)
            prompts = gather_object(prompts)

        # Build the data to log
        if self.trainer.accelerator.is_main_process:
            global_step = [str(state.global_step)] * len(prompts)
            data = list(zip(global_step, prompts, completions))
            self.table.extend(data)
            table = pd.DataFrame(columns=["step", "prompt", "completion"], data=self.table)

            if "wandb" in args.report_to:
                wandb.log({"completions": table})

            if "comet_ml" in args.report_to:
                log_table_to_comet_experiment(
                    name="completions.csv",
                    table=table,
                )

        # Save the last logged step, so we don't log the same completions multiple times
        self._last_logged_step = state.global_step


class MergeModelCallback(TrainerCallback):
    r"""
    A [`~transformers.TrainerCallback`] that merges the policy model (the model being trained) with another model based on a merge configuration.

    Args:
        merge_config ([`MergeConfig`], *optional*, defaults to `None`):
            Configuration used for the merging process. If not provided, the default [`MergeConfig`] is used.
        merge_at_every_checkpoint (`bool`, *optional*, defaults to `False`):
            Whether to merge the model at every checkpoint.
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether to push the merged model to the Hub after merging.

    Example:

    ```python
    !pip install trl[mergekit]

    from trl.mergekit_utils import MergeConfig
    from trl import MergeModelCallback

    config = MergeConfig()
    merge_callback = MergeModelCallback(config)
    trainer = DPOTrainer(..., callbacks=[merge_callback])
    ```
    """

    def __init__(
        self,
        merge_config: Optional["MergeConfig"] = None,
        merge_at_every_checkpoint: bool = False,
        push_to_hub: bool = False,
    ):
        if not is_mergekit_available():
            raise ImportError(
                "MergeModelCallback requires the `mergekit` extra. To install, run `pip install trl[mergekit]`."
            )
        self.merge_config = merge_config or MergeConfig()
        self.merge_at_every_checkpoint = merge_at_every_checkpoint
        self.push_to_hub = push_to_hub

    def _merge_and_maybe_push(self, output_dir, global_step, model):
        checkpoint_path = os.path.join(output_dir, f"checkpoint-{global_step}")
        self.merge_config.policy_model_path = checkpoint_path
        if self.merge_config.target_model_path is None:
            self.merge_config.target_model_path = model.config._name_or_path
        merge_path = os.path.join(checkpoint_path, "merged")

        merge_models(self.merge_config.create(), merge_path)

        if self.push_to_hub:
            repo_name = f"{output_dir}_checkpoint-{global_step}_merged"
            upload_model_to_hf(merge_path, repo_name)

    def on_save(self, args, state, control, model=None, **kwargs):
        if self.merge_at_every_checkpoint:
            self._merge_and_maybe_push(args.output_dir, state.global_step, model)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if not self.merge_at_every_checkpoint:
            self._merge_and_maybe_push(args.output_dir, state.global_step, model)
