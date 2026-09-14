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

import math
import random
import time
import uuid

import torch

from .async_grpo_trainer import AsyncGRPOTrainer
from .async_rollout_worker import AsyncRolloutWorker, TrainingSequence, _AsyncRolloutLoop


INTERRUPTION = "Okay, time is up. Let me stop thinking and formulate a final answer now.</think>"


class ScaleRLRolloutLoop(_AsyncRolloutLoop):
    """
    Rollout loop for the ScaleRL recipe, carrying every ingredient that is decided at generation or scoring time:
    forced length interruptions, No-Positive-Resampling, zero-variance filtering, batch-level advantage normalization
    and prompt-level loss aggregation.

    Single-turn: ScaleRL is verifiable-reasoning RL with no tools, so `_generate_one` produces one response rather than
    running the tool-calling loop.

    Args:
        think_budget (`tuple[int, int]`, *optional*, defaults to `(10240, 12288)`):
            Range the thinking budget is sampled from, per rollout.
        answer_budget (`int`, *optional*, defaults to `2048`):
            Tokens granted for the final answer after an interruption.
        pass_rate_cap (`float`, *optional*, defaults to `0.9`):
            Prompts whose historical pass rate reaches this are skipped in later epochs.
        advantage_std_decay (`float`, *optional*, defaults to `0.001`):
            EMA rate for the running advantage second moment.
    """

    def __init__(
        self,
        *args,
        think_budget=(10240, 12288),
        answer_budget=2048,
        pass_rate_cap=0.9,
        advantage_std_decay=0.001,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._think_budget = think_budget
        self._answer_budget = answer_budget
        self._pass_rate_cap = pass_rate_cap
        self._std_decay = advantage_std_decay
        self._interruption_ids = self.tokenizer.encode(INTERRUPTION, add_special_tokens=False)
        self._pass_rate: dict[int, float] = {}
        self._group_row: dict[int, int] = {}
        self._adv_sq: float | None = None

    def _repeat_iterator(self):
        """
        Cycle the dataset, skipping prompts whose historical pass rate has reached `pass_rate_cap`.

        Keyed by dataset row index, which this loop tracks per group so `_score_group` can record a pass rate against
        the prompt that produced it.
        """
        group_id = 0
        epoch = 0
        while True:
            for row_index, row in enumerate(self.dataset):
                if epoch > 0 and self._pass_rate.get(row_index, 0.0) >= self._pass_rate_cap:
                    continue
                self._group_row[group_id] = row_index
                for _ in range(self.num_generations):
                    yield group_id, row
                group_id += 1
            epoch += 1

    async def _generate_one(self, prompt, tool_dict, tools, group_id=0):
        """
        Generate one response under a sampled thinking budget, interrupting it if it has not stopped thinking.

        A generation that has not emitted `</think>` within its budget is continued from `prompt + thinking +
        [`INTERRUPTION`]`, so the model concludes instead of being cut off mid-trace. The injected phrase carries
        `completion_mask = 0`: it was not sampled from the policy, and its placeholder log-probability of `0.0` would
        otherwise make the importance ratio `exp(log_probs)` on those tokens.

        Returns:
            `tuple` of `(completion messages, completion token ids, training rows, tool calls, tool failures, rollout
            reward)`.
        """
        t_dispatch = time.monotonic()
        prompt_ids = self.tokenizer.apply_chat_template(
            prompt,
            return_dict=False,
            add_generation_prompt=True,
            chat_template=self.chat_template,
            **self.chat_template_kwargs,
        )
        think_ids, think_logprobs = await self._generate_one_turn(prompt_ids, random.randint(*self._think_budget))

        interrupted = "</think>" not in self.tokenizer.decode(think_ids)
        if interrupted:
            answer_prompt = [*prompt_ids, *think_ids, *self._interruption_ids]
            answer_ids, answer_logprobs = await self._generate_one_turn(answer_prompt, self._answer_budget)
        else:
            answer_ids, answer_logprobs = [], []
        self._push_metrics({"rollout/interrupted_frac": (float(interrupted), 1.0)})

        injected = self._interruption_ids if interrupted else []
        sequence = TrainingSequence(
            input_ids=[*prompt_ids, *think_ids, *injected, *answer_ids],
            completion_mask=[0] * len(prompt_ids) + [1] * len(think_ids) + [0] * len(injected) + [1] * len(answer_ids),
            old_log_probs=[0.0] * len(prompt_ids) + think_logprobs + [0.0] * len(injected) + answer_logprobs,
            rollout_id=uuid.uuid4().hex,
        )

        completion_ids = [*think_ids, *injected, *answer_ids]
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        self._push_rollout_metrics(
            turns=1,
            sequences=1,
            completion_ids=completion_ids,
            tally=dict.fromkeys(("clean", "realign", "fork", "transitions", "drift_tokens", "drift_max"), 0.0),
            loop_exhausted=False,
            duration_s=time.monotonic() - t_dispatch,
        )
        return [{"role": "assistant", "content": text}], completion_ids, [sequence], 0, 0, None

    async def _score_group(self, group):
        """
        Score the group, record its pass rate, drop it if its rewards are flat, and write the ScaleRL weight.

        Three of the recipe's terms end up in the one float the loss multiplies per token:

            advantage = (r_i - mean(r)) / Â_std / Σ_g |y_g|

        The first is the group-centred advantage. `Â_std` is the batch-level standard deviation, tracked as a running
        second moment over the rollout stream rather than over one assembled batch. `Σ_g |y_g|` is the group's total
        completion length, which makes each prompt contribute equally to the loss regardless of how long its
        generations ran; it is known here because this is where all `num_generations` rollouts are in hand.

        A group whose rewards are all equal has zero advantage everywhere and is dropped rather than trained on.

        Returns:
            `list[RolloutSample]`: the group's training rows, or an empty list if the group was filtered out.
        """
        samples = await super()._score_group(group)
        if not samples:
            return samples
        rewards = [s.metrics["reward"] for s in samples]
        scored = [r for r in rewards if not math.isnan(r)]
        if not scored:
            return []

        row_index = self._group_row.get(group.group_id)
        if row_index is not None:
            self._pass_rate[row_index] = sum(r > 0 for r in scored) / len(scored)

        flat = min(scored) == max(scored)
        self._push_metrics({"rollout/zero_variance_frac": (float(flat), 1.0)})
        if flat:
            return []

        mean = sum(scored) / len(scored)
        centred = [0.0 if math.isnan(r) else r - mean for r in rewards]
        for advantage in centred:
            self._adv_sq = (
                advantage**2
                if self._adv_sq is None
                else (1 - self._std_decay) * self._adv_sq + self._std_decay * advantage**2
            )
        std = math.sqrt(self._adv_sq)
        group_tokens = sum(sum(s.completion_mask) for s in samples)
        self._push_metrics({"rollout/advantage_std": std, "rollout/group_tokens": float(group_tokens)})

        for sample, advantage in zip(samples, centred, strict=True):
            sample.advantage = advantage / ((std + 1e-4) * group_tokens)
        return samples


class ScaleRLRolloutWorker(AsyncRolloutWorker):
    """Rollout worker running a [`ScaleRLRolloutLoop`] in its spawned child process."""

    _loop_cls = ScaleRLRolloutLoop


class ScaleRLTrainer(AsyncGRPOTrainer):
    """
    Trainer for the ScaleRL recipe from [The Art of Scaling Reinforcement Learning Compute for
    LLMs](https://huggingface.co/papers/2510.13786), built on [`AsyncGRPOTrainer`].

    The trainer contributes the CISPO objective; everything else the recipe specifies is decided at generation or
    scoring time and lives in [`ScaleRLRolloutLoop`], which must be passed as the `rollout_worker`. The remaining
    ingredients — PipelineRL-style off-policy generation and FP32 logits — are already how [`AsyncGRPOTrainer`] works,
    with `max_staleness` setting the off-policyness bound.

    Example:

    ```python
    from transformers import AutoTokenizer

    from trl.experimental.async_grpo import AsyncGRPOConfig, ScaleRLRolloutWorker, ScaleRLTrainer

    args = AsyncGRPOConfig(output_dir="scalerl-8b", num_generations=16, epsilon_high=5.0, max_staleness=8)
    worker = ScaleRLRolloutWorker(
        model_name=model_id,
        dataset=dataset,
        reward_funcs=[reward_correct],
        processing_class=AutoTokenizer.from_pretrained(model_id),
        num_generations=args.num_generations,
        max_inflight_tasks=256,
        max_tokens=args.max_completion_length,
        think_budget=(10240, 12288),
        answer_budget=2048,
        pass_rate_cap=0.9,
        advantage_std_decay=0.001,
    )
    trainer = ScaleRLTrainer(
        model=model_id, args=args, train_dataset=dataset, reward_funcs=[reward_correct], rollout_worker=worker
    )
    trainer.train()
    ```
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        mask_bool = inputs["attention_mask"].bool()
        input_ids = inputs["input_ids"][mask_bool].unsqueeze(0)
        completion_mask = inputs["completion_mask"][mask_bool].unsqueeze(0)
        old_log_probs = inputs["old_log_probs"][mask_bool].unsqueeze(0)
        position_ids = inputs["position_ids"][mask_bool].unsqueeze(0)
        advantages = inputs["advantages"][mask_bool].unsqueeze(0)

        forward_start = time.time()
        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids,
            labels=input_ids,
            completion_mask=completion_mask,
            use_cache=False,
        )
        log_probs, entropy = outputs["log_probs"], outputs["entropy"]
        self._last_forward_time_s = time.time() - forward_start

        completion_mask = completion_mask[:, 1:]
        old_log_probs = old_log_probs[:, 1:]
        advantages = advantages[:, 1:]
        log_ratio = log_probs - old_log_probs
        coef_1 = torch.exp(log_ratio)

        # CISPO clips the importance weight rather than the advantage-scaled objective, so no token's gradient is
        # dropped; `advantages` already carry the recipe's 1/Σ|y_g| normalizer, and `num_processes` undoes the
        # gradient averaging DDP/FSDP applies across ranks.
        clamped_ratios = torch.clamp(coef_1, max=self.epsilon_high).detach()
        per_token_loss = -clamped_ratios * advantages * log_probs
        world_size = self.accelerator.num_processes
        loss = (per_token_loss * completion_mask).sum() * world_size
        loss = loss / self.current_gradient_accumulation_steps

        if self.aux_loss_enabled:
            aux_loss = outputs["aux_loss"]
            loss = loss + self.router_aux_loss_coef * aux_loss / self.current_gradient_accumulation_steps

        with torch.no_grad():
            valid_mask = completion_mask > 0
            local_count = valid_mask.sum().float()

            # Empty masked selections sum to a 0 scalar on the right device, so no valid_mask.any() guard is needed.
            local_ratio_sum = coef_1[valid_mask].sum()
            # Approx KL: http://joschu.net/blog/kl-approx.html
            local_kl_sum = ((coef_1[valid_mask] - 1) - log_ratio[valid_mask]).sum()
            local_entropy_sum = entropy[valid_mask].sum()

            # CISPO has no trust region, so the clipping metrics of the parent's surrogate do not apply. What is
            # worth watching is how often the weight is truncated on a token that would otherwise be reinforced.
            local_clip_sum = ((coef_1 > self.epsilon_high) & (advantages > 0))[valid_mask].float().sum()

            # Batch all-reduce: [ratio_sum, kl_sum, entropy_sum, clip_sum, count]
            stats = torch.stack([local_ratio_sum, local_kl_sum, local_entropy_sum, local_clip_sum, local_count])
            stats = self.accelerator.reduce(stats, reduction="sum")
            global_ratio_sum, global_kl_sum, global_entropy_sum, global_clip_sum, global_count = stats.unbind(0)
            self._metrics["train"]["ratio"].append((global_ratio_sum / global_count).item())
            self._metrics["train"]["kl"].append((global_kl_sum / global_count).item())
            self._metrics["train"]["entropy"].append((global_entropy_sum / global_count).item())
            self._metrics["train"]["cispo_clip_ratio"].append((global_clip_sum / global_count).item())

            if self.aux_loss_enabled:
                gathered_aux = self.accelerator.reduce(aux_loss.detach().to(torch.float32), reduction="sum")
                self._metrics["train"]["aux_loss"].append((gathered_aux / world_size).item())

        # Per-step accounting, accumulated across the micro-batches of one optimizer step and flushed in
        # `training_step`. The counts are batch-wide (the collator broadcasts one value per rank), so they are read off
        # rank-local inputs without a collective. Sample rewards and packing metrics are NOT gathered here — rank 0
        # already logged them in the collator.
        n_forward_tokens = float(inputs["global_n_forward_tokens"][0])
        mean_seq_len = float(inputs["mean_seq_len"][0])
        self._step_forward_tokens += n_forward_tokens
        self._step_trained_tokens += float(inputs["global_n_tokens"][0])
        self._step_seq_len_weighted += mean_seq_len * n_forward_tokens
        self._step_samples += n_forward_tokens / mean_seq_len
        self._step_forward_s += self._last_forward_time_s
        return loss
