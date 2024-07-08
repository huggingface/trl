# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import dataclasses
import inspect
import warnings
from functools import wraps
from typing import Callable, Dict, List, Optional, Tuple, Union

import datasets
import torch
import torch.nn as nn
from accelerate.state import PartialState
from datasets import Dataset
from datasets.arrow_writer import SchemaInferenceError
from datasets.builder import DatasetGenerationError
from huggingface_hub.utils._deprecation import _deprecate_arguments
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollator,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
)
from transformers.modeling_utils import unwrap_model
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction

from ..extras.dataset_formatting import get_formatting_func_from_dataset
from ..import_utils import is_peft_available
from .sft_config import SFTConfig
from .utils import (
    ConstantLengthDataset,
    DataCollatorForCompletionOnlyLM,
    RichProgressCallback,
    neftune_post_forward_hook,
    peft_module_casting_to_bf16,
    trl_sanitze_kwargs_for_tagging,
)


if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training


class SFTTrainer(Trainer):
    r"""
    Class definition of the Supervised Finetuning Trainer (SFT Trainer).
    This class is a wrapper around the `transformers.Trainer` class and inherits all of its attributes and methods.
    The trainer takes care of properly initializing the PeftModel in case a user passes a `PeftConfig` object.

    Args:
        model (Union[`transformers.PreTrainedModel`, `nn.Module`, `str`]):
            The model to train, can be a `PreTrainedModel`, a `torch.nn.Module` or a string with the model name to
            load from cache or download. The model can be also converted to a `PeftModel` if a `PeftConfig` object is
            passed to the `peft_config` argument.
        args (Optional[`SFTConfig`]):
            The arguments to tweak for training. Will default to a basic instance of [`SFTConfig`] with the `output_dir`
            set to a directory named *tmp_trainer* in the current directory if not provided.
        data_collator (Optional[`transformers.DataCollator`]):
            The data collator to use for training.
        train_dataset (Optional[`datasets.Dataset`]):
            The dataset to use for training. We recommend users to use `trl.trainer.ConstantLengthDataset` to create their dataset.
        eval_dataset (Optional[Union[`datasets.Dataset`, Dict[`str`, `datasets.Dataset`]]]):
            The dataset to use for evaluation. We recommend users to use `trl.trainer.ConstantLengthDataset` to create their dataset.
        tokenizer (Optional[`transformers.PreTrainedTokenizer`]):
            The tokenizer to use for training. If not specified, the tokenizer associated to the model will be used.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        compute_metrics (`Callable[[transformers.EvalPrediction], Dict]`, *optional* defaults to None):
            The function used to compute metrics during evaluation. It should return a dictionary mapping metric names to metric values.
            If not specified, only the loss will be computed during evaluation.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        peft_config (`Optional[PeftConfig]`):
            The PeftConfig object to use to initialize the PeftModel.
        formatting_func (`Optional[Callable]`):
            The formatting function to be used for creating the `ConstantLengthDataset`.
    """

    _tag_names = ["trl", "sft"]

    @_deprecate_arguments(
        version="1.0.0",
        deprecated_args=[
            "dataset_text_field",
            "packing",
            "max_seq_length",
            "dataset_num_proc",
            "dataset_batch_size",
            "neftune_noise_alpha",
            "model_init_kwargs",
            "dataset_kwargs",
            "eval_packing",
            "num_of_sequences",
            "chars_per_token",
        ],
        custom_message="Deprecated positional argument(s) used in SFTTrainer, please use the SFTConfig to set these arguments instead.",
    )
    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[SFTConfig] = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
        dataset_text_field: Optional[str] = None,
        packing: Optional[bool] = False,
        formatting_func: Optional[Callable] = None,
        max_seq_length: Optional[int] = None,
        infinite: Optional[bool] = None,
        num_of_sequences: Optional[int] = 1024,
        chars_per_token: Optional[float] = 3.6,
        dataset_num_proc: Optional[int] = None,
        dataset_batch_size: int = 1000,
        neftune_noise_alpha: Optional[float] = None,
        model_init_kwargs: Optional[Dict] = None,
        dataset_kwargs: Optional[Dict] = None,
        eval_packing: Optional[bool] = None,
    ):
        if args is None:
            output_dir = "tmp_trainer"
            warnings.warn(f"No `SFTConfig` passed, using `output_dir={output_dir}`.")
            args = SFTConfig(output_dir=output_dir)
        elif args is not None and args.__class__.__name__ == "TrainingArguments":
            args_as_dict = args.to_dict()
            # Manually copy token values as TrainingArguments.to_dict() redacts them
            args_as_dict.update({k: getattr(args, k) for k in args_as_dict.keys() if k.endswith("_token")})
            args = SFTConfig(**args_as_dict)

        if model_init_kwargs is not None:
            warnings.warn(
                "You passed `model_init_kwargs` to the SFTTrainer, the value you passed will override the one in the `SFTConfig`."
            )
            args.model_init_kwargs = model_init_kwargs
        if getattr(args, "model_init_kwargs", None) is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_init_kwargs to the SFTConfig, but your model is already instantiated.")
        else:
            model_init_kwargs = args.model_init_kwargs

            torch_dtype = model_init_kwargs["torch_dtype"]
            if torch_dtype is not None:
                # Convert to `torch.dtype` if an str is passed
                if isinstance(torch_dtype, str) and torch_dtype != "auto":
                    torch_dtype = getattr(torch, torch_dtype)

                if torch_dtype != "auto" and not isinstance(torch_dtype, torch.dtype):
                    raise ValueError(
                        f"Invalid `torch_dtype` passed to the SFTConfig. Expected a string with either `torch.dtype` or 'auto', but got {torch_dtype}."
                    )

            model_init_kwargs["torch_dtype"] = torch_dtype

        if infinite is not None:
            warnings.warn(
                "The `infinite` argument is deprecated and will be removed in a future version of TRL. Use `TrainingArguments.max_steps` or `TrainingArguments.num_train_epochs` instead to control training length."
            )

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the SFTTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if packing:
            warnings.warn(
                "You passed a `packing` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`."
            )
            args.packing = packing
        if eval_packing is not None:
            warnings.warn(
                "You passed a `eval_packing` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`."
            )
            args.eval_packing = eval_packing

        if args.packing and data_collator is not None and isinstance(data_collator, DataCollatorForCompletionOnlyLM):
            raise ValueError(
                "You passed a `DataCollatorForCompletionOnlyLM` to the SFTTrainer. This is not compatible with the `packing` argument."
            )

        if is_peft_available() and peft_config is not None:
            if not isinstance(peft_config, PeftConfig):
                raise ValueError(
                    "If you want to use the PeftModel, you need to pass a PeftConfig object to the SFTTrainer."
                    f" and you passed a {type(peft_config)}."
                )

            if not isinstance(model, PeftModel):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )
                gradient_checkpointing_kwargs = getattr(args, "gradient_checkpointing_kwargs", None) or {}
                is_sharded_qlora = False
                # Below is to support QLoRA + FSDP / DS-Zero3 - one should never call
                # peft_module_casting_to_bf16 or prepare_model_for_kbit_training when doing
                # QLoRA + FSDP / DS-Zero3
                if getattr(model, "is_loaded_in_4bit", False):
                    for _, param in model.named_parameters():
                        if param.__class__.__name__ == "Params4bit":
                            is_sharded_qlora = param.data.device.type == "cpu"
                            break
                if getattr(model, "is_loaded_in_8bit", False) or (
                    getattr(model, "is_loaded_in_4bit", False) and not is_sharded_qlora
                ):
                    prepare_model_kwargs = {
                        "use_gradient_checkpointing": getattr(args, "gradient_checkpointing", False)
                    }

                    if _support_gc_kwargs:
                        prepare_model_kwargs["gradient_checkpointing_kwargs"] = gradient_checkpointing_kwargs

                    model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)

                    if args is not None:
                        args = dataclasses.replace(args, gradient_checkpointing=False)
                elif getattr(args, "gradient_checkpointing", False) and (
                    "use_reentrant" not in gradient_checkpointing_kwargs
                    or gradient_checkpointing_kwargs["use_reentrant"]
                ):
                    # For backward compatibility with older versions of transformers
                    if hasattr(model, "enable_input_require_grads"):
                        model.enable_input_require_grads()
                    else:

                        def make_inputs_require_grad(module, input, output):
                            output.requires_grad_(True)

                        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

                if (
                    "autocast_adapter_dtype" in list(inspect.signature(get_peft_model).parameters)
                    and getattr(model, "is_loaded_in_4bit", False)
                    and is_sharded_qlora
                ):
                    model = get_peft_model(model, peft_config, autocast_adapter_dtype=False)
                else:
                    model = get_peft_model(model, peft_config)
                if (
                    args is not None
                    and args.bf16
                    and getattr(model, "is_loaded_in_4bit", False)
                    and not is_sharded_qlora
                ):
                    peft_module_casting_to_bf16(model)

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
            if getattr(tokenizer, "pad_token", None) is None:
                tokenizer.pad_token = tokenizer.eos_token

        if max_seq_length is not None:
            warnings.warn(
                "You passed a `max_seq_length` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`."
            )
            args.max_seq_length = max_seq_length

        if args.max_seq_length is None:
            # to overcome some issues with broken tokenizers
            max_seq_length = min(tokenizer.model_max_length, 1024)

            warnings.warn(
                f"You didn't pass a `max_seq_length` argument to the SFTTrainer, this will default to {max_seq_length}"
            )

        if dataset_num_proc is not None:
            warnings.warn(
                "You passed a `dataset_num_proc` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`."
            )
            args.dataset_num_proc = dataset_num_proc
        self.dataset_num_proc = args.dataset_num_proc

        if dataset_batch_size != args.dataset_batch_size:
            warnings.warn(
                "You passed a `dataset_batch_size` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`."
            )
            args.dataset_batch_size = dataset_batch_size
        self.dataset_batch_size = args.dataset_batch_size

        self._trainer_supports_neftune = hasattr(args, "neftune_noise_alpha")
        if neftune_noise_alpha is not None and self._trainer_supports_neftune:
            args.neftune_noise_alpha = neftune_noise_alpha
            warnings.warn(
                "You passed a `neftune_noise_alpha` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`."
            )
            # self.neftune_noise_alpha is done at Trainer level
        elif not self._trainer_supports_neftune:
            self.neftune_noise_alpha = neftune_noise_alpha

        if dataset_text_field is not None:
            warnings.warn(
                "You passed a `dataset_text_field` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`."
            )
            args.dataset_text_field = dataset_text_field

        if formatting_func is None and args.dataset_text_field is None:
            # check if dataset has ChatML format or instruction format and is supported
            # if not stays #None
            formatting_func = get_formatting_func_from_dataset(train_dataset, tokenizer)
            # if a template is detected, we don't need to add special tokens again
            if formatting_func is not None:
                if dataset_kwargs is None:
                    dataset_kwargs = {"add_special_tokens": False}
                else:
                    dataset_kwargs["add_special_tokens"] = False

        if not args.packing:
            # If we aren't skipping data preparation, then a dataset_text_field
            # or formatting_func must be provided.
            if (
                args.dataset_text_field is None
                and formatting_func is None
                and dataset_kwargs is not None
                and "skip_prepare_dataset" in dataset_kwargs
                and dataset_kwargs["skip_prepare_dataset"]
            ):
                raise ValueError(
                    "You passed `packing=False` to the SFTTrainer/SFTConfig, but you didn't pass a `dataset_text_field` or `formatting_func` argument."
                )

            if data_collator is None:
                data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        if num_of_sequences != args.num_of_sequences:
            warnings.warn(
                "You passed a `num_of_sequences` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`."
            )
            args.num_of_sequences = num_of_sequences

        if chars_per_token != args.chars_per_token:
            warnings.warn(
                "You passed a `chars_per_token` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`."
            )
            args.chars_per_token = chars_per_token

        # Pre-process the datasets only once per node. The remaining processes will use the cache.
        with PartialState().local_main_process_first():
            if dataset_kwargs is not None:
                warnings.warn(
                    "You passed a `dataset_kwargs` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`."
                )
                args.dataset_kwargs = dataset_kwargs
            if args.dataset_kwargs is None:
                args.dataset_kwargs = {}
            if train_dataset is not None:
                train_dataset = self._prepare_dataset(
                    train_dataset,
                    tokenizer,
                    args.packing,
                    args.dataset_text_field,
                    args.max_seq_length,
                    formatting_func,
                    args.num_of_sequences,
                    args.chars_per_token,
                    remove_unused_columns=args.remove_unused_columns if args is not None else True,
                    **args.dataset_kwargs,
                )
            if eval_dataset is not None:
                _multiple = isinstance(eval_dataset, dict)
                _eval_datasets = eval_dataset if _multiple else {"singleton": eval_dataset}

                eval_packing = args.packing if args.eval_packing is None else args.eval_packing

                for _eval_dataset_name, _eval_dataset in _eval_datasets.items():
                    _eval_datasets[_eval_dataset_name] = self._prepare_dataset(
                        _eval_dataset,
                        tokenizer,
                        eval_packing,
                        args.dataset_text_field,
                        args.max_seq_length,
                        formatting_func,
                        args.num_of_sequences,
                        args.chars_per_token,
                        remove_unused_columns=args.remove_unused_columns if args is not None else True,
                        **args.dataset_kwargs,
                    )
                if not _multiple:
                    eval_dataset = _eval_datasets["singleton"]

        if tokenizer.padding_side is not None and tokenizer.padding_side != "right":
            warnings.warn(
                "You passed a tokenizer with `padding_side` not equal to `right` to the SFTTrainer. This might lead to some unexpected behaviour due to "
                "overflow issues when training a model in half-precision. You might consider adding `tokenizer.padding_side = 'right'` to your code."
            )

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

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if self.args.max_steps > 0 and args.packing:
            warnings.warn(
                "You passed `packing=True` to the SFTTrainer/SFTConfig, and you are training your model with `max_steps` strategy. The dataset will be iterated until the `max_steps` are reached."
            )
            self.train_dataset.infinite = True
        elif self.args.max_steps == -1 and args.packing:
            self.train_dataset.infinite = False

        if any(isinstance(callback, RichProgressCallback) for callback in self.callback_handler.callbacks):
            for callback in self.callback_handler.callbacks:
                # Remove the PrinterCallback to avoid duplicated prints in case we passed a `RichProgressCallback`
                if callback.__class__.__name__ == "PrinterCallback":
                    self.callback_handler.pop_callback(callback)

    @wraps(Trainer.train)
    def train(self, *args, **kwargs):
        # Activate neftune right before training.
        if self.neftune_noise_alpha is not None and not self._trainer_supports_neftune:
            self.model = self._trl_activate_neftune(self.model)

        output = super().train(*args, **kwargs)

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None and not self._trainer_supports_neftune:
            unwrapped_model = unwrap_model(self.model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                embeddings = unwrapped_model.base_model.model.get_input_embeddings()
            else:
                embeddings = unwrapped_model.get_input_embeddings()

            self.neftune_hook_handle.remove()
            del embeddings.neftune_noise_alpha

        return output

    @wraps(Trainer.push_to_hub)
    def push_to_hub(self, commit_message: Optional[str] = "End of training", blocking: bool = True, **kwargs) -> str:
        """
        Overwrite the `push_to_hub` method in order to force-add the tag "sft" when pushing the
        model on the Hub. Please refer to `~transformers.Trainer.push_to_hub` for more details.
        """
        kwargs = trl_sanitze_kwargs_for_tagging(model=self.model, tag_names=self._tag_names, kwargs=kwargs)

        return super().push_to_hub(commit_message=commit_message, blocking=blocking, **kwargs)

    def _prepare_dataset(
        self,
        dataset,
        tokenizer,
        packing,
        dataset_text_field,
        max_seq_length,
        formatting_func,
        num_of_sequences,
        chars_per_token,
        remove_unused_columns=True,
        append_concat_token=True,
        add_special_tokens=True,
        skip_prepare_dataset=False,
    ):
        if dataset is None:
            raise ValueError("The dataset should not be None")

        if skip_prepare_dataset:
            return dataset

        # If the dataset is already preprocessed (tokenized), return as-is. Only works if dataset is
        # a datasets.Dataset or datasets.IterableDataset -- not for torch Dataset
        column_names = (
            dataset.column_names if isinstance(dataset, (datasets.Dataset, datasets.IterableDataset)) else None
        )
        if column_names and "input_ids" in column_names:
            if formatting_func is not None:
                warnings.warn(
                    "You passed a dataset that is already processed (contains an `input_ids` field) together with a valid formatting function. Therefore `formatting_func` will be ignored."
                )

            return dataset

        # check if torch dataset / dataloader and do nothing
        # see https://github.com/huggingface/trl/pull/1468 for why datasets.IterableDataset needs a separate check
        if isinstance(
            dataset, (torch.utils.data.IterableDataset, torch.utils.data.Dataset, ConstantLengthDataset)
        ) and not isinstance(dataset, datasets.IterableDataset):
            return dataset

        if not packing:
            return self._prepare_non_packed_dataloader(
                tokenizer,
                dataset,
                dataset_text_field,
                max_seq_length,
                formatting_func,
                add_special_tokens,
                remove_unused_columns,
            )

        else:
            return self._prepare_packed_dataloader(
                tokenizer,
                dataset,
                dataset_text_field,
                max_seq_length,
                num_of_sequences,
                chars_per_token,
                formatting_func,
                append_concat_token,
                add_special_tokens,
            )

    def _prepare_non_packed_dataloader(
        self,
        tokenizer,
        dataset,
        dataset_text_field,
        max_seq_length,
        formatting_func=None,
        add_special_tokens=True,
        remove_unused_columns=True,
    ):
        use_formatting_func = formatting_func is not None and dataset_text_field is None
        self._dataset_sanity_checked = False

        # Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
        def tokenize(element):
            outputs = tokenizer(
                element[dataset_text_field] if not use_formatting_func else formatting_func(element),
                add_special_tokens=add_special_tokens,
                truncation=True,
                padding=False,
                max_length=max_seq_length,
                return_overflowing_tokens=False,
                return_length=False,
            )

            if use_formatting_func and not self._dataset_sanity_checked:
                if not isinstance(formatting_func(element), list):
                    raise ValueError(
                        "The `formatting_func` should return a list of processed strings since it can lead to silent bugs."
                    )
                else:
                    self._dataset_sanity_checked = True

            return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

        signature_columns = ["input_ids", "labels", "attention_mask"]

        extra_columns = list(set(dataset.column_names) - set(signature_columns))

        if not remove_unused_columns and len(extra_columns) > 0:
            warnings.warn(
                "You passed `remove_unused_columns=False` on a non-packed dataset. This might create some issues with the default collator and yield to errors. If you want to "
                f"inspect dataset other columns (in this case {extra_columns}), you can subclass `DataCollatorForLanguageModeling` in case you used the default collator and create your own data collator in order to inspect the unused dataset columns."
            )

        tokenized_dataset = dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names if remove_unused_columns else None,
            num_proc=self.dataset_num_proc,
            batch_size=self.dataset_batch_size,
        )

        return tokenized_dataset

    def _prepare_packed_dataloader(
        self,
        tokenizer,
        dataset,
        dataset_text_field,
        max_seq_length,
        num_of_sequences,
        chars_per_token,
        formatting_func=None,
        append_concat_token=True,
        add_special_tokens=True,
    ):
        if dataset_text_field is not None or formatting_func is not None:
            if tokenizer is None:
                raise ValueError("You need to pass a tokenizer when using `dataset_text_field` with `SFTTrainer`.")

            constant_length_iterator = ConstantLengthDataset(
                tokenizer,
                dataset,
                dataset_text_field=dataset_text_field,
                formatting_func=formatting_func,
                seq_length=max_seq_length,
                infinite=False,
                num_of_sequences=num_of_sequences,
                chars_per_token=chars_per_token,
                eos_token_id=tokenizer.eos_token_id,
                append_concat_token=append_concat_token,
                add_special_tokens=add_special_tokens,
            )

            if isinstance(dataset, datasets.IterableDataset):
                return constant_length_iterator

            def data_generator(constant_length_iterator):
                yield from constant_length_iterator

            try:
                packed_dataset = Dataset.from_generator(
                    data_generator, gen_kwargs={"constant_length_iterator": constant_length_iterator}
                )
            except (DatasetGenerationError, SchemaInferenceError) as exc:
                raise ValueError(
                    "Error occurred while packing the dataset. "
                    "Make sure that your dataset has enough samples to at least yield one packed sequence."
                ) from exc
            return packed_dataset
        else:
            raise ValueError(
                "You need to pass a `dataset_text_field` or `formatting_func` argument to the SFTTrainer if you want to use the `ConstantLengthDataset`."
            )

    def _trl_activate_neftune(self, model):
        r"""
        Activates the neftune as presented in this code: https://github.com/neelsjain/NEFTune and paper: https://arxiv.org/abs/2310.05914
        Since in transformers Trainer we do have an `_activate_neftune` method, we need to rename this method to avoid conflicts.
        """
        unwrapped_model = unwrap_model(model)
        if is_peft_available() and isinstance(unwrapped_model, PeftModel):
            embeddings = unwrapped_model.base_model.model.get_input_embeddings()
        else:
            embeddings = unwrapped_model.get_input_embeddings()

        embeddings.neftune_noise_alpha = self.neftune_noise_alpha
        hook_handle = embeddings.register_forward_hook(neftune_post_forward_hook)
        self.neftune_hook_handle = hook_handle
        return model
