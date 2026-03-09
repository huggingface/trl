# Copyright 2020-2026 The HuggingFace Team & Axolotl AI
# All rights reserved.
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

"""
GRPODataProducer: produces GRPO training rollouts using the transformers DataProducer protocol.

This module bridges TRL's GRPO generation pipeline with the transformers Trainer's
online-training infrastructure (``DataProducer`` / ``_OnlineEpochSource``).
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from transformers.data_producer import BaseDataProducer, ProducerConfig
from transformers.trainer_utils import seed_worker

from .utils import RepeatSampler, identity, shuffle_sequence_dict


logger = logging.getLogger(__name__)

class RolloutDataset(Dataset):
    """A ``torch.utils.data.Dataset`` wrapping the output dict from
    ``_generate_and_score_completions``.

    The output dict contains two kinds of entries:

    * **Per-sample tensors** (batch dim > 0): ``prompt_ids``, ``completion_ids``,
      ``advantages``, ``old_per_token_logps``, etc.
    * **Shared metadata** (scalar, 0-dim tensor, non-tensor, or sentinel):
      ``num_items_in_batch``, ``_pending_policy_logps``.

    ``__getitem__`` slices per-sample tensors at the requested index and passes
    shared values through unchanged.  A matching collator is created via
    :func:`make_rollout_collator`.
    """

    # Keys that are always treated as shared (not per-sample) regardless of type.
    _ALWAYS_SHARED = frozenset({"num_items_in_batch", "_pending_policy_logps"})

    def __init__(self, data: dict[str, Any]):
        self._data = data

        # Classify keys into shared vs per-sample.
        self._shared_keys: set[str] = set()
        self._sample_keys: set[str] = set()

        for key, val in data.items():
            if key in self._ALWAYS_SHARED:
                self._shared_keys.add(key)
            elif not isinstance(val, torch.Tensor):
                # Non-tensor values (lists, ints, etc.) are treated as shared.
                self._shared_keys.add(key)
            elif val.dim() == 0:
                self._shared_keys.add(key)
            else:
                self._sample_keys.add(key)

        # Determine number of samples from any per-sample tensor.
        self._num_samples = 0
        for key in self._sample_keys:
            n = data[key].size(0)
            if self._num_samples == 0:
                self._num_samples = n
            elif n != self._num_samples:
                raise ValueError(
                    f"Inconsistent sample count: key '{key}' has {n} samples, "
                    f"expected {self._num_samples}"
                )

        if self._num_samples == 0:
            raise ValueError("No per-sample tensors found in rollout data")

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item: dict[str, Any] = {}
        for key in self._sample_keys:
            item[key] = self._data[key][idx]
        for key in self._shared_keys:
            item[key] = self._data[key]
        return item


def make_rollout_collator(shared_keys: set[str]):
    """Return a collator that stacks per-sample tensors and passes shared
    keys through (taken from the first element in the batch).

    Args:
        shared_keys: Set of key names that should NOT be stacked.
    """

    def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key in batch[0]:
            if key in shared_keys:
                result[key] = batch[0][key]
            else:
                values = [item[key] for item in batch]
                if isinstance(values[0], torch.Tensor):
                    result[key] = torch.stack(values)
                else:
                    result[key] = values
        return result

    return _collate


class GRPODataProducer(BaseDataProducer):
    """Produces GRPO training rollouts using the trainer's generation pipeline.

    This producer is created *before* ``Trainer.__init__`` completes, so it
    stores only serialisable config values at construction time.  The live
    trainer reference is injected later via :meth:`set_trainer`, which also
    creates the prompt ``DataLoader``.

    Args:
        config: :class:`ProducerConfig` controlling mini-epochs, async, etc.
        prompt_dataset: The original prompt dataset (HF ``Dataset``).
        num_generations: Completions per unique prompt.
        generation_batch_size: Global generation batch size (``per_device * steps_per_gen * num_processes``).
        train_batch_size: Per-device training batch size.
        steps_per_generation: Training steps per generation round.
        shuffle_dataset: Whether to shuffle prompts.
        seed: Random seed for the prompt sampler.
    """

    def __init__(
            self,
            config: ProducerConfig,
            prompt_dataset,
            *,
            num_generations: int,
            generation_batch_size: int,
            train_batch_size: int,
            steps_per_generation: int,
            shuffle_dataset: bool,
            seed: int,
    ):
        super().__init__(config)
        self._dataset = prompt_dataset
        self._num_generations = num_generations
        self._generation_batch_size = generation_batch_size
        self._train_batch_size = train_batch_size
        self._steps_per_generation = steps_per_generation
        self._shuffle_dataset = shuffle_dataset
        self._seed = seed

        # Set later via set_trainer().
        self._trainer = None
        self._prompt_dl: DataLoader | None = None
        self._prompt_iter = None

    def set_trainer(self, trainer) -> None:
        """Inject the live trainer reference and create the prompt DataLoader.

        Must be called after ``Trainer.__init__`` completes (so that
        ``trainer.accelerator`` is available).
        """
        self._trainer = trainer
        self._init_prompt_dataloader()

    def _init_prompt_dataloader(self) -> None:
        """Create a distributed-aware prompt DataLoader using RepeatSampler.

        * ``repeat_count=1`` so each ``produce()`` call draws a fresh batch.
        * ``accelerator.prepare`` adds the ``DistributedSampler`` wrapper.
        * The dataloader is immediately removed from ``accelerator._dataloaders``
          to prevent checkpoint / memory-lifecycle interference.
        """
        trainer = self._trainer
        sampler = RepeatSampler(
            data_source=self._dataset,
            mini_repeat_count=self._num_generations,
            batch_size=self._generation_batch_size // self._num_generations,
            repeat_count=1,
            shuffle=self._shuffle_dataset,
            seed=self._seed,
        )
        dl = DataLoader(
            self._dataset,
            batch_size=self._train_batch_size * self._steps_per_generation,
            sampler=sampler,
            collate_fn=identity,
            num_workers=trainer.args.dataloader_num_workers,
            pin_memory=trainer.args.dataloader_pin_memory,
            persistent_workers=trainer.args.dataloader_persistent_workers,
            worker_init_fn=partial(
                seed_worker,
                num_workers=trainer.args.dataloader_num_workers,
                rank=trainer.args.process_index,
            ),
        )
        self._prompt_dl = trainer.accelerator.prepare(dl)

        # Don't let the accelerator track this dataloader (it's not the
        # training dataloader and shouldn't be saved/restored with checkpoints).
        acc_dls = trainer.accelerator._dataloaders
        if self._prompt_dl in acc_dls:
            acc_dls.remove(self._prompt_dl)

        self._prompt_iter = iter(self._prompt_dl)

    def _pre_produce_hook(self, inputs: list, global_step: int) -> list:
        """Called before generation to allow prompt modification.

        Override in subclasses to inject new candidates, curriculum
        prompts, or other prompt-level transformations.

        Args:
            inputs: List of prompt dicts drawn from the dataloader.
            global_step: Current training step.

        Returns:
            (Possibly modified) list of prompt dicts.
        """
        return inputs

    # -- produce -------------------------------------------------------------

    def produce(
            self,
            model: Any,
            global_step: int,
            *,
            skip_policy_logps: bool = False,
            processing_class: Any = None,
            accelerator: Any = None,
            args: Any = None,
            **kwargs,
    ) -> RolloutDataset:
        """Generate a fresh GRPO training rollout.

        1. Draw the next prompt batch from the internal prompt DataLoader.
        2. Delegate to ``trainer._generate_and_score_completions``.
        3. Shuffle the output to break prompt-group ordering.
        4. Wrap in a :class:`RolloutDataset`.

        Args:
            model: Ignored (the trainer already holds a model reference).
            global_step: Current training step.
            skip_policy_logps: When ``True``, the generation pipeline skips
                model forward passes (``old_per_token_logps``, IS ratio,
                ``ref_per_token_logps``) and sets a ``_pending_policy_logps``
                sentinel.  Used by ``AsyncDataProducer`` for background calls.
        """
        # get the next prompt batch from iterator (start over on epoch exhaustion).
        try:
            inputs = next(self._prompt_iter)
        except StopIteration:
            self._prompt_iter = iter(self._prompt_dl)
            inputs = next(self._prompt_iter)

        # Hook for subclasses to modify prompts before generation.
        inputs = self._pre_produce_hook(inputs, global_step)

        # Generate completions, compute rewards & advantages.
        output = self._trainer._generate_and_score_completions(
            inputs, skip_policy_logps=skip_policy_logps
        )

        # Strip non-sequence metadata before shuffling.  shuffle_sequence_dict
        # expects every value to be a Tensor, list, or None — plain scalars
        # (like the ``_pending_policy_logps: True`` sentinel or ``num_items_in_batch``)
        # would cause a "not subscriptable" TypeError.
        metadata = {}
        for key in list(output.keys()):
            val = output[key]
            if not isinstance(val, (torch.Tensor, list)):
                metadata[key] = output.pop(key)
            elif isinstance(val, torch.Tensor) and val.dim() == 0:
                metadata[key] = output.pop(key)

        # Shuffle to break prompt-group ordering before batching.
        # When skip_policy_logps=True (async path), we defer the shuffle to the
        # main thread — _compute_deferred_scores needs grouped (unshuffled)
        # ordering to normalise advantages per prompt group.
        if not skip_policy_logps:
            output = shuffle_sequence_dict(output)

        # When running on a background thread (skip_policy_logps=True -> async),
        # tensor creation (padding etc.) was done on this thread's CUDA stream.
        # Synchronize so all data is materialised before crossing the thread
        # boundary.
        if skip_policy_logps and torch.cuda.is_available():
            torch.cuda.synchronize()

        # Re-attach metadata that was stripped before the shuffle.
        output.update(metadata)

        return RolloutDataset(output)