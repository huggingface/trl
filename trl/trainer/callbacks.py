# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

import logging
import os

import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import gather_object, is_wandb_available
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
from transformers.utils import is_rich_available

from ..data_utils import maybe_apply_chat_template
from ..import_utils import is_mergekit_available, is_weave_available
from ..mergekit_utils import MergeConfig, merge_models, upload_model_to_hf
from ..models.utils import unwrap_model_for_generation
from .utils import get_config_model_id, log_table_to_comet_experiment


if is_rich_available():
    from rich.columns import Columns
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import Progress
    from rich.table import Table

if is_wandb_available():
    import wandb

if is_weave_available():
    import weave
    from weave import EvaluationLogger
    from weave.trace.context import weave_client_context


# Logger for module-level logging
logger = logging.getLogger(__name__)


def _generate_completions(
    prompts: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    accelerator: Accelerator,
    generation_config: GenerationConfig | None,
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
            for prompt, generation in zip(tokenized_batch.input_ids, generations, strict=True):
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
        ref_model: PreTrainedModel | torch.nn.Module,
        accelerator: Accelerator | None,
    ):
        self.accelerator = accelerator
        self.ref_model = ref_model

    @staticmethod
    def _sync_target_model(model, target_model, alpha):
        for target_param, copy_param in zip(target_model.parameters(), model.parameters(), strict=True):
            target_param.data.mul_(1.0 - alpha).add_(copy_param.data, alpha=alpha)

    @staticmethod
    def sync_target_model(model, target_model, alpha):
        deepspeed_plugin = AcceleratorState().deepspeed_plugin
        if deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3:
            import deepspeed

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
        if not is_rich_available():
            raise ImportError("RichProgressCallback requires the `rich` extra. To install, run `pip install rich`.")

        self.training_bar = None
        self.evaluation_bar = None
        self.training_task = None
        self.evaluation_task = None
        self.rich_group = None
        self.rich_console = None
        self.training_status = None
        self.current_step = None

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        self.training_bar = Progress()
        self.evaluation_bar = Progress()
        self.rich_console = Console()
        self.training_status = self.rich_console.status("Nothing to log yet ...")
        self.rich_group = Live(Panel(Group(self.training_bar, self.evaluation_bar, self.training_status)))
        self.rich_group.start()
        self.training_task = self.training_bar.add_task("[blue]Training  ", total=state.max_steps)
        self.current_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        self.training_bar.update(self.training_task, advance=state.global_step - self.current_step, update=True)
        self.current_step = state.global_step

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if not state.is_world_process_zero:
            return

        if has_length(eval_dataloader):
            if self.evaluation_task is None:
                self.evaluation_task = self.evaluation_bar.add_task("[blue]Evaluation", total=len(eval_dataloader))
            self.evaluation_bar.update(self.evaluation_task, advance=1, update=True)

    def on_evaluate(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        if self.evaluation_task is not None:
            self.evaluation_bar.remove_task(self.evaluation_task)
            self.evaluation_task = None

    def on_predict(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        if self.evaluation_task is not None:
            self.evaluation_bar.remove_task(self.evaluation_task)
            self.evaluation_task = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not (state.is_world_process_zero and self.training_bar):
            return

        # Group keys by top-level prefix
        grouped_logs = {}
        for key, value in logs.items():
            parts = key.split("/")
            group = parts[0] if len(parts) > 1 else None
            subkey = "/".join(parts[1:]) if len(parts) > 1 else key
            grouped_logs.setdefault(group, {})[subkey] = value

        # Create a table per group
        tables = []
        for group_name, metrics in grouped_logs.items():
            table = Table(
                title=f"[bold blue]{group_name}[/]" if group_name else None, header_style="bold magenta", box=None
            )
            table.add_column("Metric", justify="left", no_wrap=True)
            table.add_column("Value", justify="right")

            for metric, val in metrics.items():
                formatted = f"{val:.3f}" if isinstance(val, (float, int)) else str(val)
                table.add_row(metric, formatted)

            tables.append(Panel(table, border_style="cyan", padding=(0, 1)))

        # Arrange tables in columns using Columns
        column_layout = Columns(tables, equal=False, expand=True)
        self.training_status.update(
            Panel(column_layout, title=f"[bold green]Step {state.global_step}[/bold green]", border_style="green")
        )

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        self.rich_group.stop()
        self.training_bar = None
        self.evaluation_bar = None
        self.training_task = None
        self.evaluation_task = None
        self.rich_group = None
        self.rich_console = None
        self.training_status = None
        self.current_step = None


def _win_rate_completions_df(
    state: TrainerState, prompts: list[str], completions: list[str], winner_indices: list[str]
) -> pd.DataFrame:
    global_step = [str(state.global_step)] * len(prompts)
    data = list(zip(global_step, prompts, completions, winner_indices, strict=True))
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
        judge ([`experimental.judges.BasePairwiseJudge`]):
            The judge to use for comparing completions.
        trainer (`Trainer`):
            Trainer to which the callback will be attached. The trainer's evaluation dataset must include a `"prompt"`
            column containing the prompts for generating completions. If the `Trainer` has a reference model (via the
            `ref_model` attribute), it will use this reference model for generating the reference completions;
            otherwise, it defaults to using the initial model.
        generation_config ([`~transformers.GenerationConfig`], *optional*):
            The generation config to use for generating completions.
        num_prompts (`int`, *optional*):
            The number of prompts to generate completions for. If not provided, defaults to the number of examples in
            the evaluation dataset.
        shuffle_order (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the order of the completions before judging.
        use_soft_judge (`bool`, *optional*, defaults to `False`):
            Whether to use a soft judge that returns a win probability between 0 and 1 for the first completion vs the
            second.
    """

    def __init__(
        self,
        judge,
        trainer: Trainer,
        generation_config: GenerationConfig | None = None,
        num_prompts: int | None = None,
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
            completions = list(zip(self.ref_completions, self.ref_completions, strict=True))
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

            completions = list(zip(self.ref_completions, completions, strict=True))

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
        generation_config ([`~transformers.GenerationConfig`], *optional*):
            The generation config to use for generating completions.
        num_prompts (`int`, *optional*):
            The number of prompts to generate completions for. If not provided, defaults to the number of examples in
            the evaluation dataset.
        freq (`int`, *optional*):
            The frequency at which to log completions. If not provided, defaults to the trainer's `eval_steps`.
    """

    def __init__(
        self,
        trainer: Trainer,
        generation_config: GenerationConfig | None = None,
        num_prompts: int | None = None,
        freq: int | None = None,
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
            data = list(zip(global_step, prompts, completions, strict=True))
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


class WeaveCallback(TrainerCallback):
    r"""
    A [`~transformers.TrainerCallback`] that logs traces and evaluations to W&B Weave. The callback uses
    https://weave-docs.wandb.ai/guides/evaluation/evaluation_logger/ to log traces and evaluations at each evaluation
    step.

    Supports two modes based on the `scorers` parameter:
    - **Tracing Mode** (when scorers=None): Logs predictions for data exploration and analysis
    - **Evaluation Mode** (when scorers provided): Logs predictions with scoring and summary metrics

    Both modes use Weave's EvaluationLogger for structured, consistent data logging.

    The callback logs data during evaluation phases (`on_evaluate`) rather than training steps, making it more
    efficient and semantically correct. It gracefully handles missing weave installation by logging warnings and
    skipping weave-specific functionality. It also checks for existing weave clients before initializing new ones.

    Usage:
    ```python
    # Tracing mode (just log predictions)
    trainer = DPOTrainer(...)
    weave_callback = WeaveTraceCallback(trainer=trainer)  # project_name optional
    trainer.add_callback(weave_callback)

    # Or specify a project name
    weave_callback = WeaveTraceCallback(trainer=trainer, project_name="my-llm-training")
    trainer.add_callback(weave_callback)


    # Evaluation mode (log predictions + scores + summary)
    def accuracy_scorer(prompt: str, completion: str) -> float:
        # Your scoring logic here (metadata available via eval_attributes)
        return score


    weave_callback = WeaveTraceCallback(
        trainer=trainer,
        project_name="my-llm-training",  # optional and needed only if weave client is not initialized
        scorers={"accuracy": accuracy_scorer},
    )
    trainer.add_callback(weave_callback)
    ```

    Args:
        trainer (`Trainer`):
            Trainer to which the callback will be attached. The trainer's evaluation dataset must include a `"prompt"`
            column containing the prompts for generating completions.
        project_name (`str`, *optional*):
            Name of the Weave project where data will be logged. If not provided, will try to use existing weave client
            or fall back to the active wandb run's project name. Raises an error if none of these are available.
        scorers (`dict[str, Callable]`, *optional*):
            Dictionary mapping scorer names to scorer functions. If `None`, operates in tracing mode (predictions
            only). If provided, operates in evaluation mode (predictions + scores + summary). Scorer functions should
            have signature: `scorer(prompt: str, completion: str) -> float | int`
        generation_config ([`~transformers.GenerationConfig`], *optional*):
            Generation config to use for generating completions.
        num_prompts (`int` or `None`, *optional*):
            Number of prompts to generate completions for. If not provided, defaults to the number of examples in the
            evaluation dataset.
        dataset_name (`str`, *optional*, defaults to `"eval_dataset"`):
            Name for the dataset metadata in Weave.
        model_name (`str`, *optional*):
            Name for the model metadata in Weave. If not provided, attempts to extract from model config.
    """

    def __init__(
        self,
        trainer: Trainer,
        project_name: str | None = None,
        scorers: dict[str, callable] | None = None,
        generation_config: GenerationConfig | None = None,
        num_prompts: int | None = None,
        dataset_name: str = "eval_dataset",
        model_name: str | None = None,
    ):
        self.trainer = trainer
        self.project_name = project_name
        self.scorers = scorers or {}
        self.generation_config = generation_config
        self.dataset_name = dataset_name
        self.model_name = model_name
        self._last_logged_step = -1
        self._weave_initialized = False
        self._eval_logger = None

        if self.trainer.eval_dataset is None:
            raise ValueError("Trainer must have an evaluation dataset to use the WeaveCallback.")
        else:
            self.eval_dataset = self.trainer.eval_dataset

        if num_prompts is not None:
            self.eval_dataset = self.eval_dataset.select(range(num_prompts))

    def _initialize_weave(self):
        """Initialize Weave and EvaluationLogger if not already initialized."""
        if not self._weave_initialized:
            if not is_weave_available():
                logger.warning("Weave is not available. Please install weave to enable logging: `pip install weave`")
                return

            if wc := weave_client_context.get_weave_client():
                self._weave_client = wc
            else:
                if self.project_name is None:
                    if is_wandb_available():
                        if wandb.run is not None:
                            self.project_name = wandb.run.entity + "/" + wandb.run.project
                            logger.info(f"Using project name from active wandb run: {self.project_name}")

                    if self.project_name is None:
                        raise ValueError(
                            "No existing Weave client found and no project_name provided. "
                            "Please either initialize weave with `weave.init('project-name')`, "
                            "provide a project_name to the `WeaveTraceCallback`, "
                            "or ensure an active wandb run exists."
                        )

                self._weave_client = weave.init(self.project_name)
                logger.info(f"Initialized Weave with project: {self.project_name}")

            if self.model_name is None:
                self.model_name = getattr(self.trainer.model_wrapped.config, "_name_or_path", "unknown_model")

            self._EvaluationLogger = EvaluationLogger

            self._weave_initialized = True

    @property
    def is_evaluation_mode(self) -> bool:
        """True if scorers are provided (evaluation mode), False for tracing mode."""
        return bool(self.scorers)

    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize Weave when training begins."""
        self._initialize_weave()

    def on_evaluate(self, args, state, control, **kwargs):
        if state.global_step == self._last_logged_step:
            return

        self._initialize_weave()

        if not self._weave_initialized:
            logger.debug("Weave not initialized, skipping logging")
            return

        tokenizer = kwargs["processing_class"]
        tokenizer.padding_side = "left"
        accelerator = self.trainer.accelerator
        model = self.trainer.model_wrapped

        with accelerator.split_between_processes(self.eval_dataset["prompt"]) as prompts:
            prompts = [maybe_apply_chat_template({"prompt": prompt}, tokenizer)["prompt"] for prompt in prompts]

            completions = _generate_completions(
                prompts=prompts,
                model=model,
                tokenizer=tokenizer,
                accelerator=accelerator,
                generation_config=self.generation_config,
                batch_size=args.per_device_eval_batch_size,
            )

            all_prompts = gather_object(prompts)
            all_completions = gather_object(completions)

        if self.trainer.accelerator.is_main_process:
            eval_attributes = {
                "training_step": state.global_step,
                "model_name": self.model_name,
                "generation_config": (self.generation_config.to_dict() if self.generation_config else None),
            }

            eval_logger = self._EvaluationLogger(
                model=self.model_name,
                dataset=self.dataset_name,
                eval_attributes=eval_attributes,
            )

            successful_predictions = 0
            total_score_values = {}  # For summary statistics

            for prompt, completion in zip(all_prompts, all_completions, strict=True):
                try:
                    pred_logger = eval_logger.log_prediction(inputs={"prompt": prompt}, output=completion)

                    if self.is_evaluation_mode:
                        for scorer_name, scorer_func in self.scorers.items():
                            try:
                                score = scorer_func(prompt, completion)
                                pred_logger.log_score(scorer=scorer_name, score=score)

                                if scorer_name not in total_score_values:
                                    total_score_values[scorer_name] = []
                                total_score_values[scorer_name].append(score)

                            except Exception as scorer_e:
                                logger.warning(f"Failed to apply scorer '{scorer_name}': {scorer_e}")

                    pred_logger.finish()
                    successful_predictions += 1

                except Exception as pred_e:
                    logger.warning(f"Failed to log prediction for prompt: {pred_e}")
                    # Continue with other predictions even if one fails

            if self.is_evaluation_mode and total_score_values:
                try:
                    summary_stats = {
                        "total_predictions": len(all_prompts),
                        "successful_predictions": successful_predictions,
                    }

                    for scorer_name, scores in total_score_values.items():
                        if scores:  # Only if we have valid scores
                            summary_stats[f"avg_{scorer_name}"] = sum(scores) / len(scores)

                    eval_logger.log_summary(summary_stats)

                except Exception as summary_e:
                    logger.warning(f"Failed to log summary: {summary_e}")
            else:
                try:
                    eval_logger.finish()
                except Exception as finish_e:
                    logger.warning(f"Failed to finish evaluation logger: {finish_e}")

        self._last_logged_step = state.global_step


class MergeModelCallback(TrainerCallback):
    r"""
    A [`~transformers.TrainerCallback`] that merges the policy model (the model being trained) with another model based
    on a merge configuration.

    Args:
        merge_config ([`MergeConfig`], *optional*):
            Configuration used for the merging process. If not provided, the default [`MergeConfig`] is used.
        merge_at_every_checkpoint (`bool`, *optional*, defaults to `False`):
            Whether to merge the model at every checkpoint.
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether to push the merged model to the Hub after merging.

    Example:

    ```python
    from trl.mergekit_utils import MergeConfig
    from trl import MergeModelCallback

    config = MergeConfig()
    merge_callback = MergeModelCallback(config)
    trainer = DPOTrainer(..., callbacks=[merge_callback])
    ```
    """

    def __init__(
        self,
        merge_config: "MergeConfig | None" = None,
        merge_at_every_checkpoint: bool = False,
        push_to_hub: bool = False,
    ):
        if not is_mergekit_available():
            raise ImportError(
                "MergeModelCallback requires the `mergekit` extra. To install, run `pip install mergekit`."
            )
        self.merge_config = merge_config or MergeConfig()
        self.merge_at_every_checkpoint = merge_at_every_checkpoint
        self.push_to_hub = push_to_hub

    def _merge_and_maybe_push(self, output_dir, global_step, model):
        checkpoint_path = os.path.join(output_dir, f"checkpoint-{global_step}")
        self.merge_config.policy_model_path = checkpoint_path
        if self.merge_config.target_model_path is None:
            self.merge_config.target_model_path = get_config_model_id(model.config)
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


class BEMACallback(TrainerCallback):
    # docstyle-ignore
    r"""
    A [`~transformers.TrainerCallback`] that implements [BEMA](https://huggingface.co/papers/2508.00180)
    (Bias-Corrected Exponential Moving Average) by [Adam Block](https://huggingface.co/abblock) and [Cyril
    Zhang](https://huggingface.co/cyrilzhang). Code from https://github.com/abblock/bema under MIT license.

    BEMA computes model weights that scale like:

    $$
    \theta_t' = \alpha_t \cdot (\theta_t - \theta_0) + \text{EMA}_t
    $$

    where  \\( \theta_t \\) is the current model weights,  \\( \theta_0 \\) is a snapshot of the model weights at the
    first `update_after` step,  \\( \text{EMA}_t  \\) is the exponential moving average of the model weights, and
     \\( \alpha_t \\) is a scaling factor that decays with the number of steps  \\( t \\) as

    $$
    \alpha_t = (\rho + \gamma \cdot t)^{-\eta}.
    $$

    The EMA is computed as:

    $$
    \text{EMA}_t = (1 - \beta_t) \cdot \text{EMA}_{t-1} + \beta_t \cdot \theta_t
    $$

    where  \\( \beta_t \\) is a decay factor that decays with the number of steps  \\( t \\) as

    $$
    \beta_t = (\rho + \gamma \cdot t)^{-\kappa}.
    $$

    Args:
        update_freq (`int`, *optional*, defaults to `400`):
            Update the BEMA weights every X steps. Denoted this as  \\( \phi \\) in the paper.
        ema_power (`float`, *optional*, defaults to `0.5`):
            Power for the EMA decay factor. Denoted  \\( \kappa \\) in the paper. To disable EMA, set this to `0.0`.
        bias_power (`float`, *optional*, defaults to `0.2`):
            Power for the BEMA scaling factor. Denoted  \\( \eta \\) in the paper. To disable BEMA, set this to `0.0`.
        lag (`int`, *optional*, defaults to `10`):
            Initial offset in the weight decay schedule that controls early-stage smoothness by acting as a virtual
            starting age for the updates. Denoted as  \\( \rho \\) in the paper.
        update_after (`int`, *optional*, defaults to `0`):
            Burn-in time before starting to update the BEMA weights. Denoted  \\( \tau \\) in the paper.
        multiplier (`float`, *optional*, defaults to `1.0`):
            Initial value for the EMA decay factor. Denoted as  \\( \gamma \\) in the paper.
        min_ema_multiplier (`float`, *optional*, defaults to `0.0`):
            Minimum value for the EMA decay factor.
        device (`str`, *optional*, defaults to `"cpu"`):
            Device to use for the BEMA buffers, e.g. `"cpu"` or `"cuda"`. Note that in most cases, this device SHOULD
            BE DIFFERENT from the device used for training in order to avoid OOM.

    Example:

    ```python
    from trl import BEMACallback

    trainer = Trainer(..., callbacks=[BEMACallback()])
    ```
    """

    def __init__(
        self,
        update_freq: int = 400,
        ema_power: float = 0.5,
        bias_power: float = 0.2,
        lag: int = 10,
        update_after: int = 0,
        multiplier: float = 1.0,
        min_ema_multiplier: float = 0.0,
        device: str = "cpu",
    ):
        # User-provided hyperparams
        self.update_freq = update_freq
        self.ema_power = ema_power
        self.bias_power = bias_power
        self.lag = lag
        self.update_after = update_after
        self.multiplier = multiplier
        self.min_ema_multiplier = min_ema_multiplier
        self.device = device

        # Internal state
        self.param_names = []  # references to training model param names
        self.thetat_params = []  # references to training model params
        self.theta0_params = []  # θ₀ buffers (on self.device)
        self.ema_params = []  # EMA buffers (on self.device)
        self.running_model = None  # a copy of the model to run BEMA on

    @staticmethod
    def _unwrap_model(model):
        """
        Helper function to unwrap model from various wrappers including DataParallel, DistributedDataParallel,
        DeepSpeed, and FSDP.
        """
        # Handle DeepSpeed
        if hasattr(model, "module") and hasattr(model, "engine"):
            # DeepSpeed engine
            return model.module

        # Handle FSDP
        if hasattr(model, "_fsdp_wrapped_module"):
            # FSDP wrapped model
            return model._fsdp_wrapped_module

        # Handle DataParallel/DistributedDataParallel
        if hasattr(model, "module"):
            return model.module

        return model

    @torch.no_grad()
    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model: PreTrainedModel, **kwargs
    ):
        model = self._unwrap_model(model)

        # Create a new instance and load state_dict
        self.running_model = type(model)(model.config).to(self.device)
        self.running_model.load_state_dict(model.state_dict())

        # Cache trainable parameters once in a fixed order
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.param_names.append(name)
            self.thetat_params.append(param)

            # Clone θ₀ and EMA on the same device as model
            theta0 = param.detach().clone().to(self.device)
            self.theta0_params.append(theta0)
            self.ema_params.append(theta0.clone())  # initialize EMA with θ₀

    def _ema_beta(self, step: int) -> float:
        """Compute the EMA decay factor βₜ = (ρ + γ·t)⁻ᵏᵃᵖᵖᵃ."""
        beta = (self.lag + self.multiplier * step) ** (-self.ema_power)
        return max(beta, self.min_ema_multiplier)

    def _bema_alpha(self, step: int) -> float:
        """Compute the BEMA scaling factor αₜ = (ρ + γ·t)⁻ᵉᵗᵃ."""
        return (self.lag + self.multiplier * step) ** (-self.bias_power)

    def _update_bema_weights(self, step: int):
        beta = self._ema_beta(step)
        alpha = self._bema_alpha(step)

        # Compute EMA + BEMA in-place and write directly to running_model
        for thetat, theta0, ema, run_param in zip(
            self.thetat_params, self.theta0_params, self.ema_params, self.running_model.parameters(), strict=True
        ):
            thetat = thetat.detach().to(self.device)
            ema.mul_(1 - beta).add_(thetat, alpha=beta)  # EMA update: ema = (1 - beta) * ema + beta * θₜ
            run_param.copy_(ema + alpha * (thetat - theta0))  # BEMA update: run_param = ema + alpha * (θₜ - θ₀)

    @torch.no_grad()
    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model: PreTrainedModel, **kwargs
    ):
        step = state.global_step

        # If we haven't reached the update_after step, skip the BEMA update
        if step < self.update_after:
            return

        # Snapshot θ₀ and EMA at first update
        if step == self.update_after:
            for thetat_param, theta0_param, ema_param in zip(
                self.thetat_params, self.theta0_params, self.ema_params, strict=True
            ):
                theta0_param.copy_(thetat_param)
                ema_param.copy_(thetat_param)

        # Update BEMA weights every `update_freq` steps
        elif (step - self.update_after) % self.update_freq == 0:
            self._update_bema_weights(step)
            logger.info(f"Updated BEMA weights at step {step}")

    @torch.no_grad()
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_world_process_zero:
            save_directory = f"{args.output_dir}/bema"
            self.running_model.save_pretrained(save_directory)
            logger.info(f"Saved BEMA model to {save_directory}")
