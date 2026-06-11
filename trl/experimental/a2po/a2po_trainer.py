# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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

import inspect
import math
import textwrap
from collections import defaultdict
from collections.abc import Callable

import torch
from accelerate.logging import get_logger
from accelerate.utils import gather_object
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from ...data_utils import maybe_apply_chat_template
from ...models import create_reference_model, unwrap_model_for_generation
from ...trainer.base_trainer import _BaseTrainer
from ...trainer.utils import selective_log_softmax
from .a2po_config import A2POConfig


logger = get_logger(__name__)

# A reward function is a callable that returns a list of floats (the rewards). The callable receives prompts,
# completions, and additional columns from the dataset. To ensure forward compatibility, it should accept **kwargs.
RewardFunc = Callable[..., list[float]]


class A2POTrainer(_BaseTrainer):
    # docstyle-ignore
    """
    Trainer for the A*-PO (Optimal Advantage Regression) method, introduced in [Accelerating RL for LLM Reasoning with
    Optimal Advantage Regression](https://huggingface.co/papers/2505.20686).

    A*-PO runs in two stages:

    1. **Offline value estimation.** Before training, `num_value_samples` completions are sampled from the reference
       policy for every training prompt and scored with `reward_funcs`. The optimal value is estimated as
       `V*(x) = beta1 * log(mean_i exp(r(x, y_i) / beta1))` and cached per prompt.
    2. **On-policy regression.** During training, a single completion is generated per prompt from the current policy.
       The loss is the squared error between the implicit reward `beta2 * log(pi(y|x) / pi_ref(y|x))` and the optimal
       advantage estimate `r(x, y) - V*(x)`.

    Args:
        model (`PreTrainedModel` or `str`):
            Model to be trained, or a model identifier (string) passed to
            [`~transformers.AutoModelForCausalLM.from_pretrained`].
        reward_funcs (`Callable` or `list[Callable]`):
            Reward function(s). Each takes `prompts` and `completions` (plus dataset columns as keyword arguments) and
            returns a list of float rewards. When multiple are provided, their weighted sum (see
            [`A2POConfig.reward_weights`]) is the scalar reward `r`, which A*-PO assumes to be binary (in `{0, 1}`).
        args ([`A2POConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`], *optional*):
            Training dataset. Must contain a `"prompt"` column.
        eval_dataset ([`~datasets.Dataset`], *optional*):
            Evaluation dataset.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*):
            Processing class used to process the data. If `None`, it is loaded from the model's name with
            [`~transformers.AutoTokenizer.from_pretrained`].
        callbacks (`list[~transformers.TrainerCallback]`, *optional*):
            List of callbacks to customize the training loop.
        optimizers (`tuple[~torch.optim.Optimizer, ~torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            Tuple containing the optimizer and the learning rate scheduler.
    """

    _tag_names = ["trl", "a2po"]
    _name = "A2PO"
    _paper = {
        "title": "Accelerating RL for LLM Reasoning with Optimal Advantage Regression",
        "id": "2505.20686",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @article{brantley2025accelerating,
                title        = {{Accelerating RL for LLM Reasoning with Optimal Advantage Regression}},
                author       = {Kiant\'e Brantley and Mingyu Chen and Zhaolin Gao and Jason D. Lee and Wen Sun and Wenhao Zhan and Xuezhou Zhang},
                year         = 2025,
                eprint       = {arXiv:2505.20686},
            }"""),
    }

    def __init__(
        self,
        model: PreTrainedModel | str,
        reward_funcs: RewardFunc | list[RewardFunc],
        args: A2POConfig | None = None,
        train_dataset=None,
        eval_dataset=None,
        processing_class: PreTrainedTokenizerBase | None = None,
        callbacks=None,
        optimizers=(None, None),
    ):
        # Args
        if args is None:
            args = A2POConfig(f"{model if isinstance(model, str) else model.config._name_or_path}-A2PO")

        # Models
        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model, **(args.model_init_kwargs or {}))
        model_id = model.config._name_or_path

        # Some models (e.g. SmolVLM/Idefics3) don't support the `logits_to_keep` argument and error out if we pass it.
        # Inspect the forward method so Stage 2 can pass the argument only when it is supported.
        self.model_kwarg_keys = inspect.signature(model.forward).parameters.keys()

        # Reference model: a frozen copy of the initial policy. Stage 1 samples from it and Stage 2 regularizes to it.
        self.ref_model = create_reference_model(model)

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model_id, padding_side="left")

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_funcs = reward_funcs
        if args.reward_weights is None:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)
        else:
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)

        # Generation: N samples per prompt in Stage 1, a single sample in Stage 2
        self.value_generation_config = GenerationConfig(
            max_new_tokens=args.max_completion_length,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_return_sequences=args.num_value_samples,
            pad_token_id=processing_class.pad_token_id,
        )
        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_completion_length,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_return_sequences=1,
            pad_token_id=processing_class.pad_token_id,
        )

        # Optimal values V*(x), keyed by prompt text. Populated lazily by Stage 1 at the start of training.
        self._optimal_values: dict[str, float] | None = None

        # Metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

        # The data collator returns the list of features untouched; generation happens in `_prepare_inputs`.
        def data_collator(features):
            return features

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    def _calculate_rewards(self, prompts, completions, **reward_kwargs):
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, reward_func in enumerate(self.reward_funcs):
            output = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
            rewards_per_func[:, i] = torch.tensor(output, dtype=torch.float32, device=device)
        # A*-PO assumes the (weighted) total reward is binary in {0, 1}.
        return (rewards_per_func * self.reward_weights.to(device)).sum(dim=1)

    def _get_sequence_logps(self, model, input_ids, attention_mask, logits_to_keep):
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        # Only pass `logits_to_keep` when the model supports it (some models and VLMs don't).
        if "logits_to_keep" in self.model_kwarg_keys:
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            model_inputs["logits_to_keep"] = logits_to_keep + 1
        logits = model(**model_inputs).logits
        logits = logits[:, :-1, :]  # the last logit predicts beyond the sequence
        # Only keep the last logits_to_keep. For models that support logits_to_keep, this is a no-op.
        logits = logits[:, -logits_to_keep:, :]
        completion_ids = input_ids[:, -logits_to_keep:]
        per_token_logps = selective_log_softmax(logits, completion_ids)
        completion_mask = attention_mask[:, -logits_to_keep:]
        return (per_token_logps * completion_mask).sum(dim=1)

    # Stage 1: offline optimal value estimation
    def _estimate_optimal_values(self):
        beta1 = self.args.beta1
        n = self.args.num_value_samples
        optimal_values = {}
        all_incorrect = set()

        # Stage 2 looks up V* for every prompt it scores, including eval prompts, so estimate over both datasets.
        datasets = [self.train_dataset] if self.eval_dataset is None else [self.train_dataset, self.eval_dataset]
        for dataset in datasets:
            dataloader = self.accelerator.prepare(
                DataLoader(dataset, batch_size=self.args.per_device_train_batch_size, collate_fn=list)
            )
            for batch in dataloader:
                prompts = [example["prompt"] for example in batch]
                prompts_text = [
                    maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in batch
                ]

                # Sample N completions per prompt from the reference policy
                inputs = self.processing_class(
                    prompts_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.args.max_prompt_length,
                    add_special_tokens=False,
                ).to(self.accelerator.device)
                with unwrap_model_for_generation(self.ref_model, self.accelerator) as ref_model:
                    completion_ids = ref_model.generate(**inputs, generation_config=self.value_generation_config)

                prompt_length = inputs["input_ids"].size(1)
                completions_text = self.processing_class.batch_decode(
                    completion_ids[:, prompt_length:], skip_special_tokens=True
                )

                # Each prompt is repeated N times by `num_return_sequences`. Forward any extra dataset columns (e.g.
                # "solution") to the reward functions, repeated to align with the N samples per prompt.
                keys = [key for key in batch[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in batch for _ in range(n)] for key in keys}
                repeated_prompts = [p for p in prompts for _ in range(n)]
                rewards = self._calculate_rewards(repeated_prompts, completions_text, **reward_kwargs)
                rewards = rewards.view(len(prompts), n)

                # V*(x) = beta1 * log(mean_i exp(r_i / beta1)), computed stably
                v_star = beta1 * (torch.logsumexp(rewards / beta1, dim=1) - math.log(n))

                for j, prompt_text in enumerate(prompts_text):
                    optimal_values[prompt_text] = v_star[j].item()
                    if rewards[j].sum() == 0:
                        all_incorrect.add(prompt_text)

        # Each rank estimates V* for its shard of prompts; share them so any rank can score any prompt.
        # `gather_object` concatenates the per-rank lists, so pass flat lists and rebuild on every rank.
        self._optimal_values = dict(gather_object(list(optimal_values.items())))
        all_incorrect = set(gather_object(list(all_incorrect)))

        # Drop training prompts whose reference samples all scored zero (no learning signal). Eval prompts are kept
        # so evaluation can still look up their V*.
        if self.args.filter_all_incorrect:
            self.train_dataset = self.train_dataset.filter(
                lambda example: maybe_apply_chat_template(example, self.processing_class)["prompt"]
                not in all_incorrect
            )
        logger.info(f"Stage 1 complete: estimated V* for {len(self._optimal_values)} prompts.")

    # Stage 2: on-policy regression
    def _prepare_inputs(self, inputs):
        # Estimate V* on first use, e.g. when `evaluate()` is called without a preceding `train()`.
        if self._optimal_values is None:
            self._estimate_optimal_values()
        mode = "train" if self.model.training else "eval"
        device = self.accelerator.device
        prompts = [example["prompt"] for example in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

        # One on-policy completion per prompt
        prompt_inputs = self.processing_class(
            prompts_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.args.max_prompt_length,
            add_special_tokens=False,
        ).to(device)
        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)
        prompt_length = prompt_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # Scalar (binary) reward and cached optimal value. Forward extra dataset columns to the reward functions.
        keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
        rewards = self._calculate_rewards(prompts, completions_text, **reward_kwargs)
        v_star = torch.tensor([self._optimal_values[p] for p in prompts_text], dtype=torch.float32, device=device)

        # Attention mask: the tokenizer's prompt mask followed by the completion mask. The completion mask is 1 up to
        # and including the first EOS and 0 afterwards, so the terminal EOS stays in the log-prob sum (a plain
        # `!= pad_token_id` mask would drop it when `pad_token == eos_token`).
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        attention_mask = torch.cat([prompt_inputs["attention_mask"], completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        with torch.no_grad():
            ref_logps = self._get_sequence_logps(self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep)

        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["v_star"].append(v_star.mean().item())

        return {
            "input_ids": prompt_completion_ids,
            "attention_mask": attention_mask,
            "logits_to_keep": logits_to_keep,
            "ref_logps": ref_logps,
            "rewards": rewards,
            "v_star": v_star,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        mode = "train" if self.model.training else "eval"

        policy_logps = self._get_sequence_logps(
            model, inputs["input_ids"], inputs["attention_mask"], inputs["logits_to_keep"]
        )

        # Implicit reward beta2 * log(pi / pi_ref), regressed onto the optimal advantage r - V*
        implicit_reward = self.args.beta2 * (policy_logps - inputs["ref_logps"])
        target = inputs["rewards"] - inputs["v_star"]
        loss = ((implicit_reward - target) ** 2).mean()

        self._metrics[mode]["implicit_reward"].append(implicit_reward.mean().item())
        self._metrics[mode]["advantage"].append(target.mean().item())
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: list[str] | None = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def train(self, *args, **kwargs):
        if self._optimal_values is None:
            logger.info("Running Stage 1: offline optimal value estimation...")
            self._estimate_optimal_values()
        return super().train(*args, **kwargs)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()
