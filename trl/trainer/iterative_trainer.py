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
from typing import Callable, List, Optional

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from datasets import Dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast

from ..core import PPODecorators, set_seed
from . import IterativeConfig, RunningMoments


class IterativeTrainer:
    """
    The IterativeTrainer can be used to finetune models with methods that require some steps between optimization.

    Attributes:
        **config** (`IterativeConfig`) -- Configuration object for IterativeTrainer.
        **model** (`PreTrainedModel`) -- Model to be optimized, Hugging Face transformer model with a causal language modeling head.
            Check the documentation of `PreTrainedModelWrapper` for more details.
        **tokenizer** (`PreTrainedTokenizerBase`) -- Tokenizer to be used for encoding the
            data. Check the documentation of `transformers.PreTrainedTokenizer` and
            `transformers.PreTrainedTokenizerFast` for more details.
        **optimizer** (`torch.optim.Optimizer`, *optional*) -- Optimizer to be used for training. If no optimizer is
            provided, the trainer will create an Adam optimizer with the learning rate specified in the configuration
            object.
        **data_collator** (DataCollatorForLanguageModeling, *optional*) -- Data collator to be used for training and
            passed along the dataloader.
    """

    def __init__(
        self,
        config: IterativeConfig = None,
        model: PreTrainedModel = None,
        tokenizer: PreTrainedTokenizerBase = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        data_collator: Optional[Callable] = None,
    ):
        """
        Initialize IterativeTrainer.

        Args:
            config (`IterativeConfig`):
                Configuration object for IterativeTrainer.
            model (`PreTrainedModel`):
                Hugging Face transformer model.
            tokenizer (`transformers.PreTrainedTokenizerBase`):
                Hugging Face tokenizer
            optimizer (Optional[`torch.optim.Optimizer`]):
                Optimizer used for training. If `None`, the `Adam` is used as default.
            data_collator (Optional[function]):
                Data collator function.
        """

        super().__init__(config)

        # initial seed for reproducible experiments
        set_seed(config.seed)

        # Step 0: check positional arguments validity
        if not isinstance(config, IterativeConfig):
            raise ValueError(f"config must be a IterativeConfig, got {type(config)}")
        if not isinstance(tokenizer, (PreTrainedTokenizerBase)):
            raise ValueError(
                f"tokenizer must be a PreTrainedTokenizerBase like a PreTrainedTokenizer or a PreTrainedTokenizerFast, got {type(tokenizer)}"
            )

        # Step 1: Initialize Accelerator
        self.accelerator = Accelerator(
            log_with=config.log_with,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            project_config=ProjectConfiguration(**config.project_kwargs),
            **config.accelerator_kwargs,
        )

        is_using_tensorboard = config.log_with is not None and config.log_with == "tensorboard"

        self.accelerator.init_trackers(
            config.tracker_project_name,
            config=dict(trl_ppo_trainer_config=config.to_dict()) if not is_using_tensorboard else config.to_dict(),
            init_kwargs=config.tracker_kwargs,
        )

        self.model = model
        self.model_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.is_encoder_decoder = hasattr(self.model, "is_encoder_decoder")
        self.is_peft_model = getattr(self.model, "is_peft_model", False)

        if not (isinstance(tokenizer, PreTrainedTokenizer) or isinstance(tokenizer, PreTrainedTokenizerFast)):
            raise ValueError(
                "tokenizer must be a transformers.PreTrainedTokenizer or transformers.PreTrainedTokenizerFast"
            )
        self.tokenizer = tokenizer

        self.data_collator = data_collator

        if optimizer is None:
            self.optimizer = Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config.learning_rate,
            )
        else:
            self.optimizer = optimizer

        (self.model, self.optimizer, self.data_collator,) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.data_collator,
        )

        self.is_distributed = self.accelerator.distributed_type == "MULTI_GPU"

        # post process for PP
        if not getattr(self.model, "is_sequential_parallel", False):
            self.current_device = self.accelerator.device
        else:
            self.current_device = torch.device("cuda:0")

        PPODecorators.optimize_cuda_cache = self.config.optimize_cuda_cache

        self.running = RunningMoments(self.accelerator)
        self.is_encoder_decoder = hasattr(self.model, "is_encoder_decoder")

    def prepare_model_inputs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor):
        if self.is_encoder_decoder:
            input_data = self.data_collator(
                [
                    {"input_ids": i, "attention_mask": a, "labels": l}
                    for i, a, l in zip(input_ids, attention_mask, labels)
                ]
            ).to(self.model.device)

        else:
            input_data = self.data_collator(
                [{"input_ids": ids, "attention_mask": torch.ones_like(ids)} for ids in labels]
            ).to(self.model.device)

            input_data.pop("decoder_input_ids", None)  # This is directly computed inside the model

            return input_data

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Loss is computed as in the HuggingFace Trainer.

        Subclass and override for custom behavior.
        """

        outputs = model(**inputs)

        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    @PPODecorators.empty_cuda_cache()
    def step(
        self, input_ids: List[torch.LongTensor], attention_mask: List[torch.LongTensor], labels: List[torch.LongTensor]
    ):
        """
        Run an optimisation step given a list of input_ids, attention_mask, and labels.
        Args:
            input_ids (List[`torch.LongTensor`]):
                List of tensors containing the input_ids
            attention_mask (List[`torch.LongTensor`]):
                List of tensors containing the attenton_mask
            labels (List[`torch.FloatTensor`]):
                List of tensors containing the labels (if set to None, will default to input_ids)
        Returns:
            `dict[str, Any]`: A summary of the training statistics
        """

        self.model.train()
        model_inputs = self.prepare_model_inputs(input_ids, attention_mask, labels)

        model_inputs_names = list(model_inputs.keys())

        batch_dict = {}
        batch_dict.update(model_inputs)

        def collator(data):
            return_dict = dict()
            for key in data[0]:
                if key in ["input_ids", "attention_mask", "labels"]:
                    return_dict[key] = torch.stack([d[key] for d in data]).to(self.model.device)
            return return_dict

        batch_data = Dataset.from_dict(batch_dict)
        batch_data.set_format("torch")

        step_dataloader = DataLoader(
            batch_data,
            batch_size=self.config.step_batch_size,
            shuffle=True,
            collate_fn=collator,
        )

        all_stats = []

        for _, batch in enumerate(step_dataloader):
            with self.accelerator.accumulate(self.model):
                model_inputs = {k: batch[k] for k in model_inputs_names}
                loss = self.compute_loss(self.model, model_inputs)

                self.accelerator.backward(loss)

                self.optimizer.step()
                self.optimizer.zero_grad()

                # update stats etc
                all_stats.append(dict(loss=dict(total=loss.detach())))

        return all_stats
