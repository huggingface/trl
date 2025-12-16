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

import pandas as pd
from accelerate import Accelerator
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

from ..models.utils import unwrap_model_for_generation
from ..trainer.utils import log_table_to_comet_experiment


if is_wandb_available():
    import wandb

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
    # TODO: Override model.generation_config with generation_kwargs
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
    from trl import DPOTrainer
    from trl.experimental.judges import PairRMJudge
    from trl.experimental.winrate_callback import WinRateCallback

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
