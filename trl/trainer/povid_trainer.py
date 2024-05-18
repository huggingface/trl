import inspect
import random
import warnings
from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import is_deepspeed_available
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput

from ..import_utils import is_peft_available, is_wandb_available
from ..models import PreTrainedModelWrapper, create_reference_model
from .utils import DPODataCollatorWithPadding, disable_dropout_in_model, pad_to_length

if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed


class POVIDTrainer(DPOTrainer):
    """
    A custom trainer class for training models with POVID.

    Args:
        model (Union[PreTrainedModel, nn.Module, str]): The model to train.
        ref_model (Optional[Union[PreTrainedModel, nn.Module, str]]): The reference model.
        beta (float, optional): The beta factor in DPO loss. Defaults to 0.1.
        loss_type (Literal["sigmoid", "hinge"], optional): The type of DPO loss to use. Defaults to "sigmoid".
        args (TrainingArguments): The arguments to use for training.
        data_collator (Optional[DataCollator], optional): The data collator to use for training. Defaults to None.
        label_pad_token_id (int, optional): The label pad token id. Defaults to -100.
        padding_value (int, optional): The padding value. Defaults to 0.
        truncation_mode (str, optional): The truncation mode to use. Defaults to "keep_end".
        train_dataset (Optional[Dataset], optional): The dataset to use for training. Defaults to None.
        eval_dataset (Optional[Union[Dataset, Dict[str, Dataset]]], optional): The dataset to use for evaluation. Defaults to None.
        tokenizer (Optional[PreTrainedTokenizerBase], optional): The tokenizer to use for training. Defaults to None.
        model_init (Optional[Callable[[], PreTrainedModel]], optional): The model initializer to use for training. Defaults to None.
        callbacks (Optional[List[TrainerCallback]], optional): The callbacks to use for training. Defaults to None.
        optimizers (Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR], optional): The optimizer and scheduler to use for training. Defaults to (None, None).
        preprocess_logits_for_metrics (Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]], optional): The function to use to preprocess the logits before computing the metrics. Defaults to None.
        max_length (Optional[int], optional): The maximum length of the sequences in the batch. Defaults to None.
        max_prompt_length (Optional[int], optional): The maximum length of the prompt. Defaults to None.
        max_target_length (Optional[int], optional): The maximum length of the target. Defaults to None.
        stage (int, optional): The stage of the training. Defaults to 1.
        peft_config (Optional[Dict], optional): The PEFT configuration to use for training. Defaults to None.
        is_encoder_decoder (Optional[bool], optional): Whether the model is an encoder-decoder. Defaults to None.
        disable_dropout (bool, optional): Whether to disable dropouts in model and ref_model. Defaults to True.
        generate_during_eval (bool, optional): Whether to sample and log generations during evaluation step. Defaults to False.
        compute_metrics (Optional[Callable[[EvalLoopOutput], Dict]], optional): The function to use to compute the metrics. Defaults to None.
        model_init_kwargs (Optional[Dict], optional): Optional kwargs to pass when instantiating the model from a string. Defaults to None.
        ref_model_init_kwargs (Optional[Dict], optional): Optional kwargs to pass when instantiating the ref model from a string. Defaults to None.
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, str] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        beta: float = 0.1,
        loss_type: Literal["sigmoid", "hinge"] = "sigmoid",
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        stage: int = 1,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        model_init_kwargs: Optional[Dict] = None,
        ref_model_init_kwargs: Optional[Dict] = None,
    ):
        if model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_kwargs to the DPOTrainer. But your model is already instantiated.")

        if ref_model_init_kwargs is None:
            ref_model_init_kwargs = {}
        elif not isinstance(ref_model, str):
            raise ValueError(
                "You passed ref_model_kwargs to the DPOTrainer. But your ref_model is already instantiated."
            )

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the DPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if isinstance(ref_model, str):
            warnings.warn(
                "You passed a ref model_id to the DPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM`"
            )
            ref_model = AutoModelForCausalLM.from_pretrained(ref_model, **ref_model_init_kwargs)

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_config, we merge and unload it first
            if isinstance(model, PeftModel):
                pass

            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )

                preprare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                if _support_gc_kwargs:
                    preprare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                model = prepare_model_for_kbit_training(model, **preprare_model_kwargs)
            elif getattr(args, "gradient_checkpointing", False):
                # For backward compatibility with older versions of transformers
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:

                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)

                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            # get peft model with the given config
            model = get_peft_model(model, peft_config)

        elif getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if generate_during_eval and not is_wandb_available():
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases to be installed."
                " Please install `wandb` to resolve."
            )

        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif is_encoder_decoder is None:
            raise ValueError("When no model is provided, you need to pass the parameter is_encoder_decoder.")
        else:
            self.is_encoder_decoder = is_encoder_decoder

        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model:
            # The `model` with adapters turned off will be used as the reference model
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(model)

        if data_collator is None:
            if tokenizer is None:
                raise ValueError(
                    "max_length or a tokenizer must be specified when using the default DPODataCollatorWithPadding"
                )
            if max_length is None:
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `max_length` in the DPOTrainer's init"
                    " it will be set to `512` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                max_length = 512
            if max_prompt_length is None:
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `max_prompt_length` in the DPOTrainer's init"
                    " it will be set to `128` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                max_prompt_length = 128

            if max_target_length is None and self.is_encoder_decoder:
                warnings.warn(
                    "When using DPODataCollatorWithPadding with an encoder decoder architecture, you should set `max_target_length` in the DPOTrainer's init"
                    " it will be set to `128` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                max_target_length = 128

            data_collator = DPODataCollatorWithPadding(
                tokenizer,
                max_length=max_length,
                max_prompt_length=max_prompt_length,
                label_pad_token_id=label_pad_token_id,
                padding_value=padding_value,
                truncation_mode=truncation_mode,
                is_encoder_decoder=self.is_encoder_decoder,
                max_target_length=max_target_length,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        if disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.max_length = max_length
        self.generate_during_eval = generate_during_eval
        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value

        self.beta = beta
        self.loss_type = loss_type
        self.stage = stage

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        if self.ref_model is None:
            if not hasattr(self.accelerator.unwrap_model(self.model), "disable_adapter"):
                raise ValueError(
                    "You are using a `peft` version that does not support `disable_adapter`. Please update your `peft` version to the latest version."
                )
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)


    def concatenated_inputs(self, batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
        """
        Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch (Dict[str, Union[List, torch.LongTensor]]): A batch of data.

        Returns:
            Dict[str, torch.LongTensor]: A dictionary containing the concatenated inputs.
        """
        concatenated_batch = {}

        if self.is_encoder_decoder:
            max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
        else:
            max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k or self.is_encoder_decoder else self.padding_value
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k or self.is_encoder_decoder else self.padding_value
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(self.accelerator.device)

        if self.is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1)
            concatenated_batch["concatenated_attention_mask"] = batch["prompt_attention_mask"].repeat(2, 1)

        return concatenated_batch

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_free: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps (torch.FloatTensor): Log probabilities of the policy model for the chosen responses.
            policy_rejected_logps (torch.FloatTensor): Log probabilities of the policy model for the rejected responses.
            reference_chosen_logps (torch.FloatTensor): Log probabilities of the reference model for the chosen responses.
            reference_rejected_logps (torch.FloatTensor): Log probabilities of the reference model for the rejected responses.
            reference_free (bool, optional): If True, ignore the provided reference model. Defaults to False.

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]: A tuple containing the losses, chosen rewards, and rejected rewards.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        if reference_free:
            ref_logratios = 0

        logits = pi_logratios - ref_logratios

        if self.loss_type == "sigmoid":
            losses = -F.logsigmoid(self.beta * logits)
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge']")

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards

    def _get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """
        Compute the log probabilities of the given labels under the given logits.

        Args:
            logits (torch.FloatTensor): Logits of the model.
            labels (torch.LongTensor): Labels for which to compute the log probabilities.
            average_log_prob (bool, optional): If True, return the average log probability per token. Defaults to False.

        Returns:
            torch.FloatTensor: A tensor containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not self.is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != self.label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == self.label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def _get_noisy_batch_logps(
        self,
        logits: torch.FloatTensor,
        chosen_logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """
        Compute the log probabilities of the given labels under the given logits.

        Args:
            logits (torch.FloatTensor): Logits of the model.
            chosen_logits (torch.FloatTensor): Logits of the chosen model.
            labels (torch.LongTensor): Labels for which to compute the log probabilities.
            average_log_prob (bool, optional): If True, return the average log probability per token. Defaults to False.

        Returns:
            torch.FloatTensor: A tensor containing the average/sum log probabilities of the given labels under the given logits.
        """
        if not self.is_encoder_decoder:
            new_chosen_logits = chosen_logits[:, :-1, :]
            logits = logits[:, :-1, :]
            labels = labels[:, 1:].clone()
        loss_mask = labels != self.label_pad_token_id

        _, max_indices = torch.max(logits, dim=2)
        per_token_logps = torch.gather(new_chosen_logits.log_softmax(-1), dim=2, index=max_indices.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        Args:
            model (nn.Module): The model to run.
            batch (Dict[str, Union[List, torch.LongTensor]]): The batch of inputs.

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]: The log probabilities and logits for the chosen and rejected inputs.
        """
        concatenated_batch = self.concatenated_inputs(batch)
        len_chosen = batch["chosen_labels"].shape[0]
        chosen_batch = concatenated_batch["concatenated_input_ids"][:len_chosen]
        rejected_batch = concatenated_batch["concatenated_input_ids"][len_chosen:]
        chosen_mask = concatenated_batch["concatenated_attention_mask"][:len_chosen]
        rejected_mask = concatenated_batch["concatenated_attention_mask"][len_chosen:]
        chosen_label = concatenated_batch["concatenated_labels"][:len_chosen]
        rejected_label = concatenated_batch["concatenated_labels"][len_chosen:]
        chosen_model_kwargs = (
            {
                "labels": chosen_label,
                "decoder_input_ids": concatenated_batch.pop("chosen_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )
        rejected_model_kwargs = (
            {
                "labels": rejected_label,
                "decoder_input_ids": concatenated_batch.pop("rejected_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )
        chosen_logits = model(
            input_ids=chosen_batch,
            labels=chosen_label,
            images=batch['images'],
            attention_mask=chosen_mask,
            **chosen_model_kwargs,
        ).logits.to(torch.float32)

        _, _, _, _, _, new_chosen_labels = self.model.prepare_inputs_labels_for_multimodal(
            input_ids=chosen_batch,
            position_ids=None,
            attention_mask=chosen_mask,
            past_key_values=None,
            labels=chosen_label,
            images=batch['images']
        )

        chosen_logps = self._get_batch_logps(
            chosen_logits,
            new_chosen_labels,
            average_log_prob=False,
        )

        rejected_logits = model(
            input_ids=rejected_batch,
            labels=rejected_label,
            images=batch['images'],
            attention_mask=rejected_mask,
            **rejected_model_kwargs,
        ).logits.to(torch.float32)

        _, _, _, _, _, new_rejected_labels = self.model.prepare_inputs_labels_for_multimodal(
            input_ids=rejected_batch,
            position_ids=None,
            attention_mask=rejected_mask,
            past_key_values=None,
            labels=rejected_label,
            images=batch['images']
        )

        rejected_logps = self._get_batch_logps(
            rejected_logits,
            new_rejected_labels,
            average_log_prob=False,
        )

        return chosen_logps, rejected_logps, chosen_logits, rejected_logits

    def concatenated_stage2_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Run the given model on the given batch of inputs for stage 2 training, concatenating the chosen and rejected inputs together.

        Args:
            model (nn.Module): The model to run.
            batch (Dict[str, Union[List, torch.LongTensor]]): The batch of inputs.

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]: The log probabilities and logits for the chosen and rejected inputs.
        """
        concatenated_batch = self.concatenated_inputs(batch)
        len_chosen = batch["chosen_labels"].shape[0]
        chosen_batch = concatenated_batch["concatenated_input_ids"][:len_chosen]
        rejected_batch = concatenated_batch["concatenated_input_ids"][len_chosen:]
        chosen_mask = concatenated_batch["concatenated_attention_mask"][:len_chosen]
        rejected_mask = concatenated_batch["concatenated_attention_mask"][len_chosen:]
        chosen_label = concatenated_batch["concatenated_labels"][:len_chosen]
        rejected_label = concatenated_batch["concatenated_labels"][len_chosen:]
        chosen_model_kwargs = (
            {
                "labels": chosen_label,
                "decoder_input_ids": concatenated_batch.pop("chosen_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )
        rejected_model_kwargs = (
            {
                "labels": rejected_label,
                "decoder_input_ids": concatenated_batch.pop("rejected_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )
        chosen_logits = model(
            input_ids=chosen_batch,
            labels=chosen_label,
            images=batch['images'],
            attention_mask=chosen_mask,
            **chosen_model_kwargs,
        ).logits.to(torch.float32)

        _, _, _, _, _, new_chosen_labels = self.model.prepare_inputs_labels_for_multimodal(
            input_ids=chosen_batch,
            position_ids=None,
            attention_mask=chosen_mask,
            past_key_values=None,
            labels=chosen_label,
            images=batch['images']
        )

        chosen_logps = self._get_batch_logps(
            chosen_logits,
            new_chosen_labels,
            average_log_prob=False,
        )

        rejected_logits_own = model(
            input_ids=chosen_batch,
            labels=chosen_label,
            images=batch['images_noisy'],
            attention_mask=chosen_mask,
            **chosen_model_kwargs,
        ).logits.to(torch.float32)

        _, _, _, _, _, new_rejected_labels = self.model.prepare_inputs_labels_for_multimodal(
            input_ids=rejected_batch,
            position_ids=None,
            attention_mask=rejected_mask,
            past_key_values=None,
            labels=rejected_label,
            images=batch['images_noisy']
        )
        rejected_logits = model(
            input_ids=rejected_batch,
            labels=rejected_label,
            images=batch['images'],
            attention_mask=rejected_mask,
            **rejected_model_kwargs,
        ).logits.to(torch.float32)

        rejected_logps = self._get_batch_logps(
            rejected_logits,
            new_rejected_labels,
            average_log_prob=False,
        )
        rejected_own_logps = self._get_noisy_batch_logps(
            rejected_logits_own,
            chosen_logits,
            new_chosen_labels,
            average_log_prob=False,
        )

        return chosen_logps, rejected_logps, chosen_logits, rejected_logits, rejected_own_logps

    def get_batch_metrics(
        self,
        model: nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple[torch.FloatTensor, Dict[str, float]]:
        """
        Compute the DPO loss and other metrics for the given batch of inputs for train or test.

        Args:
            model (nn.Module): The model to run.
            batch (Dict[str, Union[List, torch.LongTensor]]): The batch of inputs.
            train_eval (Literal["train", "eval"], optional): Whether the mode is train or eval. Defaults to "train".

        Returns:
            Tuple[torch.FloatTensor, Dict[str, float]]: The loss and metrics.
        """
        metrics = {}
        if self.stage == 1:
            policy_chosen_logps, policy_rejected_logps, policy_chosen_logits, policy_rejected_logits = self.concatenated_forward(model, batch)
            with torch.no_grad():
                if self.ref_model is None:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        reference_chosen_logps, reference_rejected_logps, _, _ = self.concatenated_forward(self.model, batch)
                else:
                    reference_chosen_logps, reference_rejected_logps, _, _ = self.concatenated_forward(self.ref_model, batch)

            losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
            )
            reward_accuracies = (chosen_rewards > rejected_rewards).float()
        elif self.stage == 2:
            policy_chosen_logps, policy_rejected_logps, policy_chosen_logits, policy_rejected_logits, rejected_own_logps = self.concatenated_stage2_forward(model, batch)
            with torch.no_grad():
                if self.ref_model is None:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        reference_chosen_logps, reference_rejected_logps, _, _, reference_rejected_own_logps = self.concatenated_stage2_forward(self.model, batch)
                else:
                    reference_chosen_logps, reference_rejected_logps, _, _, reference_rejected_own_logps = self.concatenated_stage2_forward(self.ref_model, batch)

            self_losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                policy_chosen_logps,
                rejected_own_logps,
                reference_chosen_logps,
                reference_rejected_own_logps,
            )

            losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
            )
            losses = 0.5 * self_losses + 0.5 * losses
            reward_accuracies = (chosen_rewards > rejected_rewards).float()
        else:
            raise ValueError("Stage must be 1 or 2.")
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().mean()

        return losses.mean(), metrics

