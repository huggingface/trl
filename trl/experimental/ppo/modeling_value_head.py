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

import json
import logging
import os

import torch
import torch.nn as nn
from accelerate import PartialState
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import (
    EntryNotFoundError,
    HFValidationError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
)
from safetensors.torch import load_file as safe_load_file
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    PreTrainedModel,
    is_torch_npu_available,
    is_torch_xpu_available,
)
from transformers.utils import is_peft_available


if is_peft_available():
    from peft import (
        PeftConfig,
        PeftModel,
        PeftModelForCausalLM,
        PeftModelForSeq2SeqLM,
        PromptLearningConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )


class PreTrainedModelWrapper(nn.Module):
    """
    Wrapper for a [`~transformers.PreTrainedModel`] implemented as a standard PyTorch [`torch.nn.Module`].

    This class provides a compatibility layer that preserves the key attributes and methods of the original
    [`~transformers.PreTrainedModel`], while exposing a uniform interface consistent with PyTorch modules. It enables
    seamless integration of pretrained Transformer models into custom training, evaluation, or inference workflows.

    Attributes:
        pretrained_model ([`~transformers.PreTrainedModel`]):
            The model to be wrapped.
        parent_class ([`~transformers.PreTrainedModel`]):
            The parent class of the model to be wrapped.
        supported_args (`list`):
            The list of arguments that are supported by the wrapper class.
    """

    transformers_parent_class = None
    supported_args = None
    supported_modules = ("v_head",)
    supported_rm_modules = ("score",)
    supported_pretrained_model_architectures = (
        (PreTrainedModel)
        if not is_peft_available()
        else (PreTrainedModel, PeftModelForCausalLM, PeftModelForSeq2SeqLM)
    )

    def __init__(
        self, pretrained_model=None, score_module=None, supports_rm_adapter=False, rm_adapter_name=None, **kwargs
    ):
        super().__init__()
        self.pretrained_model = pretrained_model

        self.config = pretrained_model.config
        self.prepare_inputs_for_generation = pretrained_model.prepare_inputs_for_generation
        self.is_loaded_in_8bit = getattr(pretrained_model, "is_loaded_in_8bit", False)
        self.is_loaded_in_4bit = getattr(pretrained_model, "is_loaded_in_4bit", False)
        self.is_sequential_parallel = False

        if hasattr(pretrained_model, "gradient_checkpointing_disable"):
            self.gradient_checkpointing_disable = pretrained_model.gradient_checkpointing_disable

        if hasattr(pretrained_model, "gradient_checkpointing_enable"):
            self.gradient_checkpointing_enable = pretrained_model.gradient_checkpointing_enable

        if hasattr(pretrained_model, "enable_input_require_grads"):
            self.enable_input_require_grads = pretrained_model.enable_input_require_grads

        self.supports_rm_adapter = supports_rm_adapter
        self.rm_adapter_name = rm_adapter_name
        self.policy_adapter_name = "default"
        if score_module is not None:
            self.score = score_module

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Instantiates a new model from a pretrained model from `transformers`. The pretrained model is loaded using the
        `from_pretrained` method of the [`~transformers.PreTrainedModel`] class. The arguments that are specific to the
        [`~transformers.PreTrainedModel`] class are passed along this method and filtered out from the `kwargs`
        argument.

        Args:
            pretrained_model_name_or_path (`str` or [`~transformers.PreTrainedModel`]):
                The path to the pretrained model or its name.
            *model_args (`list`, *optional*):
                Additional positional arguments passed along to the underlying model's `from_pretrained` method.
            **kwargs (`dict`, *optional*):
                Additional keyword arguments passed along to the underlying model's `from_pretrained` method. We also
                pre-process the kwargs to extract the arguments that are specific to the
                [`~transformers.PreTrainedModel`] class and the arguments that are specific to trl models. The kwargs
                also support `prepare_model_for_kbit_training` arguments from `peft` library.
        """
        if kwargs is not None:
            peft_config = kwargs.pop("peft_config", None)
            reward_adapter = kwargs.pop("reward_adapter", None)
            reward_adapter_name = kwargs.pop("reward_adapter_name", "reward_adapter")
            is_trainable = kwargs.pop("is_trainable", False)
            trl_model_args, pretrained_kwargs, peft_quantization_kwargs = cls._split_kwargs(kwargs)
            token = pretrained_kwargs.get("token", None)
        else:
            peft_config = None
            is_trainable = False
            trl_model_args = {}
            pretrained_kwargs = {}
            peft_quantization_kwargs = {}
            token = None

        if reward_adapter is not None and not isinstance(reward_adapter, str):
            raise ValueError(
                "The `reward_adapter` argument should be a string representing the name of local path or the Hub id to the Reward Modeling adapter."
            )

        is_peft_model = False

        current_device = cls._get_current_device()
        if isinstance(pretrained_model_name_or_path, str):
            quantization_config = pretrained_kwargs.get("quantization_config", None)
            if quantization_config is not None:
                is_loaded_in_8bit = getattr(quantization_config, "load_in_8bit", False)
                is_loaded_in_4bit = getattr(quantization_config, "load_in_4bit", False)
            else:
                is_loaded_in_8bit = pretrained_kwargs["load_in_8bit"] if "load_in_8bit" in pretrained_kwargs else False
                is_loaded_in_4bit = pretrained_kwargs["load_in_4bit"] if "load_in_4bit" in pretrained_kwargs else False
        else:
            is_loaded_in_8bit = getattr(pretrained_model_name_or_path, "is_loaded_in_8bit", False)
            is_loaded_in_4bit = getattr(pretrained_model_name_or_path, "is_loaded_in_4bit", False)

        if (is_loaded_in_8bit or is_loaded_in_4bit) and "device_map" not in pretrained_kwargs:
            # warn users
            logging.warning(
                "The `device_map` argument is not provided. We will override the device_map argument."
                " to set the entire"
                " model on the current device. If you want to set the model on multiple devices, please provide"
                " a custom `device_map` argument."
            )
            pretrained_kwargs["device_map"] = {"": current_device}

        if is_peft_available() and peft_config is not None and not isinstance(peft_config, PeftConfig):
            raise ValueError("The `peft_config` argument should be an instance of `peft.PeftConfig` class.")

        # First, load the pre-trained model using the parent-class
        # either `AutoModelForCausalLM` or `AutoModelForSeq2SeqLM`
        if isinstance(pretrained_model_name_or_path, str):
            if is_peft_available():
                try:
                    # If there is a trained peft adapter in the hub, load its config.
                    remote_adapter_config = hf_hub_download(
                        pretrained_model_name_or_path,
                        "adapter_config.json",
                        token=token,
                    )
                except (EntryNotFoundError, LocalEntryNotFoundError, HFValidationError, RepositoryNotFoundError):
                    remote_adapter_config = None
            else:
                remote_adapter_config = None

            local_adapter_present = os.path.exists(os.path.join(pretrained_model_name_or_path, "adapter_config.json"))

            if (local_adapter_present or remote_adapter_config is not None) and is_peft_available():
                if peft_config is not None:
                    logging.warning(
                        "`peft_config` argument ignored since a peft config file was found in "
                        f"{pretrained_model_name_or_path}"
                    )

                # Load the trained peft adapter config
                if local_adapter_present:
                    trained_adapter_config = PeftConfig.from_pretrained(pretrained_model_name_or_path)
                else:
                    remote_adapter_dir = os.path.dirname(remote_adapter_config)
                    trained_adapter_config = PeftConfig.from_pretrained(remote_adapter_dir)

                # Load the pretrained base model
                pretrained_model = cls.transformers_parent_class.from_pretrained(
                    trained_adapter_config.base_model_name_or_path, *model_args, **pretrained_kwargs
                )

                # Wrap the pretrained model with the trained peft adapter
                pretrained_model = PeftModel.from_pretrained(
                    pretrained_model, pretrained_model_name_or_path, is_trainable=is_trainable, token=token
                )
                logging.info("Trained peft adapter loaded")
            else:
                pretrained_model = cls.transformers_parent_class.from_pretrained(
                    pretrained_model_name_or_path, *model_args, **pretrained_kwargs
                )

                if peft_config is not None:
                    # Initialize a new peft adapter with the given config
                    if is_loaded_in_8bit or is_loaded_in_4bit:
                        pretrained_model = prepare_model_for_kbit_training(
                            pretrained_model,
                            **peft_quantization_kwargs,
                        )
                    pretrained_model = get_peft_model(pretrained_model, peft_config)
                    logging.info("peft adapter initialised")

        elif isinstance(pretrained_model_name_or_path, cls.supported_pretrained_model_architectures):
            pretrained_model = pretrained_model_name_or_path

            if peft_config is not None and isinstance(pretrained_model, PreTrainedModel):
                # Initialize a new peft adapter with the given config
                if is_loaded_in_8bit or is_loaded_in_4bit:
                    pretrained_model = prepare_model_for_kbit_training(
                        pretrained_model,
                        **peft_quantization_kwargs,
                    )
                pretrained_model = get_peft_model(pretrained_model, peft_config)
                logging.info("peft adapter initialised")
        else:
            raise ValueError(
                "pretrained_model_name_or_path should be a string or a PreTrainedModel, "
                f"but is {type(pretrained_model_name_or_path)}"
            )

        if is_peft_available():
            if isinstance(pretrained_model, PeftModel):
                is_peft_model = True
                # for backward compatibility
                if hasattr(pretrained_model, "active_peft_config") and isinstance(
                    pretrained_model.active_peft_config, PromptLearningConfig
                ):
                    raise ValueError("PromptLearningConfig is not supported for PPO training.")

        # Add reward modeling adapter if specified
        if not is_peft_model and reward_adapter is not None:
            raise ValueError("reward_adapter can only be used with a PeftModel. ")
        elif is_peft_model and reward_adapter is not None:
            score_module = cls.add_and_load_reward_modeling_adapter(
                pretrained_model, reward_adapter, reward_adapter_name, token=token
            )
            multi_adapter_args = {
                "score_module": score_module,
                "supports_rm_adapter": True,
                "rm_adapter_name": reward_adapter_name,
            }
        else:
            multi_adapter_args = {"supports_rm_adapter": False}

        # Then, create the full model by instantiating the wrapper class
        model = cls(pretrained_model, **multi_adapter_args, **trl_model_args)

        # if resume_training, load the state_dict again - this is ok since the
        # state_dict is removed from the model after loading it.
        is_resuming_training = True
        if isinstance(pretrained_model_name_or_path, str):
            safe_filename = os.path.join(pretrained_model_name_or_path, "model.safetensors")
            filename = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")

            sharded_index_filename = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin.index.json")
            safe_sharded_index_filename = os.path.join(pretrained_model_name_or_path, "model.safetensors.index.json")
            is_sharded = False
            use_safe = os.path.exists(safe_filename)

            if not (os.path.exists(filename) or os.path.exists(safe_filename)):
                # Try with `pytorch_model.bin`
                filename, files_to_download, is_sharded, is_resuming_training = cls._get_checkpoint_from_hub(
                    pretrained_model,
                    pretrained_model_name_or_path,
                    sharded_index_filename,
                    token=token,
                )
                # Try with safetensors
                if filename is None and files_to_download is None:
                    safe_filename, files_to_download, is_sharded, is_resuming_training = cls._get_checkpoint_from_hub(
                        pretrained_model,
                        pretrained_model_name_or_path,
                        safe_sharded_index_filename,
                        token=token,
                        model_name="model.safetensors",
                        model_index_name="model.safetensors.index.json",
                    )
                    use_safe = True
                else:
                    use_safe = False

            loading_func = safe_load_file if use_safe else torch.load
            load_kwargs = {} if use_safe else {"map_location": "cpu", "weights_only": True}

            if is_resuming_training:
                if is_sharded:
                    # download each file and add it to the state_dict
                    state_dict = {}

                    for shard_file in files_to_download:
                        filename = hf_hub_download(
                            pretrained_model_name_or_path,
                            shard_file,
                            token=token,
                        )
                        state_dict.update(loading_func(filename, **load_kwargs))
                else:
                    state_dict = loading_func(filename if not use_safe else safe_filename, **load_kwargs)

        else:
            state_dict = pretrained_model_name_or_path.state_dict()

        model.is_peft_model = is_peft_model
        model.current_device = current_device

        if is_resuming_training:
            model.post_init(state_dict=state_dict)

        return model

    @classmethod
    def _get_checkpoint_from_hub(
        cls,
        pretrained_model,
        pretrained_model_name_or_path,
        index_filename,
        token=None,
        model_name="pytorch_model.bin",
        model_index_name="pytorch_model.bin.index.json",
    ):
        files_to_download = None
        filename = None
        is_resuming_training = True
        is_sharded = False

        try:
            filename = hf_hub_download(
                pretrained_model_name_or_path,
                model_name,
                token=token,
            )
        # sharded
        except (EntryNotFoundError, LocalEntryNotFoundError, HFValidationError, RepositoryNotFoundError):
            if os.path.exists(index_filename):
                index_file_name = index_filename
            else:
                try:
                    index_file_name = hf_hub_download(
                        pretrained_model_name_or_path,
                        model_index_name,
                        token=token,
                    )
                except (EntryNotFoundError, LocalEntryNotFoundError, HFValidationError, RepositoryNotFoundError):
                    # not continue training, do not have v_head weight
                    is_resuming_training = False
                    logging.warning(
                        f"A {type(pretrained_model)} model is loaded from '{pretrained_model_name_or_path}', "
                        f"and no v_head weight is found. This IS expected if you are not resuming PPO training."
                    )
            # load json
            if is_resuming_training:
                with open(index_file_name) as f:
                    index = json.load(f)
                # check filename with `v_head` or any known extra module:
                files_to_download = set()
                for k, v in index["weight_map"].items():
                    if any(module in k for module in cls.supported_modules):
                        files_to_download.add(v)
                is_sharded = True

        return filename, files_to_download, is_sharded, is_resuming_training

    @classmethod
    def _get_current_device(cls):
        r"""
        Get the current device. For GPU & XPU, we return the local process index using the `accelerate.PartialState`
        object to handle corner cases when running scripts in distributed environments.

        Returns:
            current_device (`int | str`):
                The current device.
        """
        state = PartialState()
        if torch.cuda.is_available() or is_torch_xpu_available():
            return state.local_process_index
        elif is_torch_npu_available():
            return f"npu:{state.local_process_index}"
        else:
            return "cpu"

    @classmethod
    def _split_kwargs(cls, kwargs):
        """
        Separate the kwargs from the arguments that we support inside `supported_args` and the ones that we don't.
        """
        check_peft_kwargs = False

        if is_peft_available():
            from peft import prepare_model_for_kbit_training

            check_peft_kwargs = True

        supported_kwargs = {}
        unsupported_kwargs = {}
        peft_kwargs = {}

        for key, value in kwargs.items():
            if key in cls.supported_args:
                supported_kwargs[key] = value
            else:
                unsupported_kwargs[key] = value

            if check_peft_kwargs:
                if key in prepare_model_for_kbit_training.__code__.co_varnames:
                    peft_kwargs[key] = value
                    if key in unsupported_kwargs:
                        unsupported_kwargs.pop(key)

        return supported_kwargs, unsupported_kwargs, peft_kwargs

    @classmethod
    def add_and_load_reward_modeling_adapter(
        cls, pretrained_model, adapter_model_id, adapter_name="reward_model_adapter", token=None
    ):
        r"""
        Add and load a reward modeling adapter. This method can only be used if the model is a `PeftModel` and if you
        have initialized the model with the `reward_modeling_adapter_id` argument, pointing to the id of the reward
        modeling adapter. The latest needs also to contain the score head in order to produce the reward.
        """
        pretrained_model.load_adapter(adapter_model_id, adapter_name, is_trainable=False)
        pretrained_model.train()

        filename = os.path.join(adapter_model_id, "adapter_model.bin")
        safe_loading = False
        if not os.path.exists(filename):
            try:
                local_filename = hf_hub_download(
                    adapter_model_id,
                    "adapter_model.bin",
                    token=token,
                )
            except Exception:
                filename = os.path.join(adapter_model_id, "adapter_model.safetensors")
                safe_loading = True
                if not os.path.exists(filename):
                    try:
                        local_filename = hf_hub_download(
                            adapter_model_id,
                            "adapter_model.safetensors",
                            token=token,
                        )
                    except Exception as exc:
                        raise ValueError(
                            "Could not find adapter model in the Hub, make sure you have the correct adapter model id."
                        ) from exc
                else:
                    local_filename = filename
        else:
            local_filename = filename

        loading_func = safe_load_file if safe_loading else torch.load
        load_kwargs = {} if safe_loading else {"map_location": "cpu", "weights_only": True}

        adapter_state_dict = loading_func(local_filename, **load_kwargs)

        for score_name_candidate in cls.supported_rm_modules:
            if any(score_name_candidate in name for name in adapter_state_dict.keys()):
                score_name = score_name_candidate
                # we have found the correct head name and can break
                break

        score_dict = {}

        for name, param in adapter_state_dict.items():
            if score_name in name:
                key_name = ".".join(name.split(".")[-1:])
                score_dict[key_name] = param.to(cls._get_current_device())

        num_labels, hidden_dim = score_dict["weight"].shape
        has_bias = any("bias" in name for name in adapter_state_dict.keys())

        score = nn.Linear(hidden_dim, num_labels, bias=has_bias).to(
            device=cls._get_current_device(),
            dtype=pretrained_model.dtype,
        )
        score.load_state_dict(score_dict)
        for param in score.parameters():
            param.requires_grad = False

        return score

    def push_to_hub(self, *args, **kwargs):
        r"""
        Push the pretrained model to the hub. This method is a wrapper around
        [`~transformers.PreTrainedModel.push_to_hub`]. Please refer to the documentation of
        [`~transformers.PreTrainedModel.push_to_hub`] for more information.

        Args:
            *args (`list`, *optional*):
                Positional arguments passed along to the underlying model's `push_to_hub` method.
            **kwargs (`dict`, *optional*):
                Keyword arguments passed along to the underlying model's `push_to_hub` method.
        """
        raise NotImplementedError

    def save_pretrained(self, *args, **kwargs):
        r"""
        Save the pretrained model to a directory. This method is a wrapper around
        [`~transformers.PreTrainedModel.save_pretrained`]. Please refer to the documentation of
        [`~transformers.PreTrainedModel.save_pretrained`] for more information.

        Args:
            *args (`list`, *optional*):
                Positional arguments passed along to the underlying model's `save_pretrained` method.
            **kwargs (`dict`, *optional*):
                Keyword arguments passed along to the underlying model's `save_pretrained` method.
        """
        state_dict = kwargs.get("state_dict")
        if state_dict is None:
            state_dict = self.state_dict()
            kwargs["state_dict"] = state_dict

        # if it is a peft model only save the `v_head` state_dict and
        # pop the `state_dict` from the kwargs to avoid silent bugs with `peft`
        if self.is_peft_model:
            save_path = args[0]
            save_path = os.path.join(save_path, "pytorch_model.bin")
            torch.save(state_dict, save_path)
            _ = kwargs.pop("state_dict", None)

        return self.pretrained_model.save_pretrained(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        r"""
        Return the state_dict of the pretrained model.
        """
        raise NotImplementedError

    def post_init(self, *args, **kwargs):
        r"""
        Post initialization method. This method is called after the model is instantiated and loaded from a checkpoint.
        It can be used to perform additional operations such as loading the state_dict.
        """
        raise NotImplementedError

    def compute_reward_score(self, input_ids, attention_mask=None, **kwargs):
        r"""
        Computes the reward score for a given input. The method has first to enable the adapter and then compute the
        reward score. After that the model disables the reward modeling adapter and enables the default ppo adapter
        again.
        """
        if not self.supports_rm_adapter:
            raise ValueError("This model does not support reward modeling adapter.")

        # enable rm adapter
        self.pretrained_model.set_adapter(self.rm_adapter_name)
        self.pretrained_model.eval()

        with torch.no_grad():
            base_model_output = self.pretrained_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )

            last_hidden_states = base_model_output.hidden_states[-1]
            scores = self.score(last_hidden_states)

        self.pretrained_model.set_adapter(self.policy_adapter_name)
        self.pretrained_model.eval()

        return scores


class ValueHead(nn.Module):
    r"""
    The ValueHead class implements a head for GPT2 that returns a scalar for each output token.
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        if not hasattr(config, "summary_dropout_prob"):
            summary_dropout_prob = kwargs.pop("summary_dropout_prob", 0.1)
        else:
            summary_dropout_prob = config.summary_dropout_prob

        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()

        # some models such as OPT have a projection layer before the word embeddings - e.g. OPT-350m
        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        if hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
        elif hasattr(config, "is_encoder_decoder"):
            if config.is_encoder_decoder and hasattr(config, "decoder"):
                if hasattr(config.decoder, "hidden_size"):
                    hidden_size = config.decoder.hidden_size

        self.summary = nn.Linear(hidden_size, 1)

        self.flatten = nn.Flatten()

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)

        # For now force upcast in fp32 if needed. Let's keep the
        # output in fp32 for numerical stability.
        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)

        output = self.summary(output)
        return output


class AutoModelForCausalLMWithValueHead(PreTrainedModelWrapper):
    """
    An autoregressive model with a value head in addition to the language model head. This class inherits from
    [`experimental.ppo.PreTrainedModelWrapper`] and wraps a [`~transformers.PreTrainedModel`] class. The wrapper class
    supports classic functions such as `from_pretrained`, `push_to_hub` and `generate`. To call a method of the wrapped
    model, simply manipulate the `pretrained_model` attribute of this class.

    Class attributes:
        - **transformers_parent_class** ([`~transformers.PreTrainedModel`]) -- The parent class of the wrapped model.
          This
            should be set to `transformers.AutoModelForCausalLM` for this class.
        - **supported_args** (`tuple`) -- A tuple of strings that are used to identify the arguments that are supported
            by the [`ValueHead`] class. Currently, the supported args are:
            - **summary_dropout_prob** (`float`, `optional`, defaults to `None`) -- The dropout probability for the
                [`ValueHead`] class.
            - **v_head_initializer_range** (`float`, `optional`, defaults to `0.2`) -- The initializer range for the
                [`ValueHead`] if a specific initialization strategy is selected.
            - **v_head_init_strategy** (`str`, `optional`, defaults to `None`) -- The initialization strategy for the
                [`ValueHead`]. Currently, the supported strategies are:
                - **`None`** -- Initializes the weights of the [`ValueHead`] with a random distribution. This is the
                  default strategy.
                - **"normal"** -- Initializes the weights of the [`ValueHead`] with a normal distribution.
    """

    transformers_parent_class = AutoModelForCausalLM
    supported_args = (
        "summary_dropout_prob",
        "v_head_initializer_range",
        "v_head_init_strategy",
    )

    def __init__(self, pretrained_model, **kwargs):
        """
        Initializes the model.

        Args:
            pretrained_model ([`~transformers.PreTrainedModel`]):
                The model to wrap. It should be a causal language model such as GPT2. or any model mapped inside the
                `AutoModelForCausalLM` class.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the [`ValueHead`] class.
        """
        super().__init__(pretrained_model, **kwargs)
        v_head_kwargs, _, _ = self._split_kwargs(kwargs)
        self.v_head = ValueHead(self.pretrained_model.config, **v_head_kwargs)
        self._init_weights(**v_head_kwargs)

    def _init_weights(self, **kwargs):
        r"""
        Initializes the weights of the value head. The default initialization strategy is random. Users can pass a
        different initialization strategy by passing the `v_head_init_strategy` argument when calling
        `.from_pretrained`. Supported strategies are:
        - `normal`: initializes the weights with a normal distribution.

        Args:
            **kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the [`ValueHead`] class. These arguments can contain
                the `v_head_init_strategy` argument as well as the `v_head_initializer_range` argument.
        """
        initializer_range = kwargs.pop("v_head_initializer_range", 0.2)
        # random init by default
        init_strategy = kwargs.pop("v_head_init_strategy", None)
        if init_strategy is None:
            # do nothing
            pass
        elif init_strategy == "normal":
            self.v_head.summary.weight.data.normal_(mean=0.0, std=initializer_range)
            self.v_head.summary.bias.data.zero_()

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        return_past_key_values=False,
        **kwargs,
    ):
        r"""
        Applies a forward pass to the wrapped model and returns the logits of the value head.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, `optional`):
                Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
                (see `past_key_values` input) to speed up sequential decoding.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            return_past_key_values (bool): A flag indicating if the computed hidden-states should be returned.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the wrapped model.
        """
        kwargs["output_hidden_states"] = True  # this had already been set in the LORA / PEFT examples
        kwargs["past_key_values"] = past_key_values

        if self.is_peft_model and self.pretrained_model.active_peft_config.peft_type == "PREFIX_TUNING":
            kwargs.pop("past_key_values")

        base_model_output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        last_hidden_state = base_model_output.hidden_states[-1]
        lm_logits = base_model_output.logits
        loss = base_model_output.loss

        if last_hidden_state.device != self.v_head.summary.weight.device:
            last_hidden_state = last_hidden_state.to(self.v_head.summary.weight.device)

        value = self.v_head(last_hidden_state).squeeze(-1)

        # force upcast in fp32 if logits are in half-precision
        if lm_logits.dtype != torch.float32:
            lm_logits = lm_logits.float()

        if return_past_key_values:
            return (lm_logits, loss, value, base_model_output.past_key_values)
        else:
            return (lm_logits, loss, value)

    def generate(self, *args, **kwargs):
        r"""
        A simple wrapper around the `generate` method of the wrapped model. Please refer to the
        [`generate`](https://huggingface.co/docs/transformers/internal/generation_utils) method of the wrapped model
        for more information about the supported arguments.

        Args:
            *args (`list`, *optional*):
                Positional arguments passed to the `generate` method of the wrapped model.
            **kwargs (`dict`, *optional*):
                Keyword arguments passed to the `generate` method of the wrapped model.
        """
        return self.pretrained_model.generate(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        r"""
        Returns the state dictionary of the model. We add the state dictionary of the value head to the state
        dictionary of the wrapped model by prepending the key with `v_head.`.
        """
        if not self.is_peft_model:
            pretrained_model_state_dict = self.pretrained_model.state_dict(*args, **kwargs)
        else:
            # if it is a peft model, only save the v_head
            pretrained_model_state_dict = {}

        v_head_state_dict = self.v_head.state_dict(*args, **kwargs)
        for k, v in v_head_state_dict.items():
            pretrained_model_state_dict[f"v_head.{k}"] = v
        return pretrained_model_state_dict

    def push_to_hub(self, *args, **kwargs):
        self.pretrained_model.v_head = self.v_head

        return self.pretrained_model.push_to_hub(*args, **kwargs)

    def post_init(self, state_dict):
        r"""
        We add the state dictionary of the value head to the state dictionary of the wrapped model by prepending the
        key with `v_head.`. This function removes the `v_head.` prefix from the keys of the value head state
        dictionary.
        """
        for k in list(state_dict.keys()):
            if "v_head." in k:
                state_dict[k.replace("v_head.", "")] = state_dict.pop(k)
        self.v_head.load_state_dict(state_dict, strict=False)
        del state_dict

        if hasattr(self.pretrained_model, "hf_device_map"):
            if (
                "cpu" in self.pretrained_model.hf_device_map.values()
                or "disk" in self.pretrained_model.hf_device_map.values()
            ):
                raise ValueError(
                    "The model is offloaded on CPU or disk - CPU & disk offloading is not supported for ValueHead models."
                )

            first_device = list(set(self.pretrained_model.hf_device_map.values()))[0]
            if isinstance(first_device, int):
                if is_torch_npu_available():
                    first_device = f"npu:{first_device}"
                elif is_torch_xpu_available():
                    first_device = f"xpu:{first_device}"
                else:
                    first_device = f"cuda:{first_device}"
            self.v_head = self.v_head.to(first_device)

            def set_device_hook(module, input, outputs):
                new_output = ()
                for output in outputs:
                    if isinstance(output, torch.Tensor):
                        new_output += (output.to(first_device),)
                    else:
                        new_output += (output,)
                return new_output

            self.register_forward_hook(set_device_hook)

            self.is_sequential_parallel = True


class AutoModelForSeq2SeqLMWithValueHead(PreTrainedModelWrapper):
    """
    A seq2seq model with a value head in addition to the language model head. This class inherits from
    [`experimental.ppo.PreTrainedModelWrapper`] and wraps a [`~transformers.PreTrainedModel`] class. The wrapper class
    supports classic functions such as `from_pretrained` and `push_to_hub` and also provides some additional
    functionalities such as `generate`.

    Args:
        pretrained_model ([`~transformers.PreTrainedModel`]):
            The model to wrap. It should be a causal language model such as GPT2. or any model mapped inside the
            [`~transformers.AutoModelForSeq2SeqLM`] class.
        kwargs:
            Additional keyword arguments passed along to the [`ValueHead`] class.
    """

    transformers_parent_class = AutoModelForSeq2SeqLM
    lm_head_namings = ["lm_head", "embed_out", "output_projection"]
    supported_args = (
        "summary_dropout_prob",
        "v_head_initializer_range",
        "v_head_init_strategy",
    )

    def __init__(self, pretrained_model, **kwargs):
        super().__init__(pretrained_model, **kwargs)
        v_head_kwargs, _, _ = self._split_kwargs(kwargs)
        self.is_encoder_decoder = True

        if not self._has_lm_head():
            raise ValueError("The model does not have a language model head, please use a model that has one.")

        self.v_head = ValueHead(self.pretrained_model.config, **v_head_kwargs)

        self._init_weights(**v_head_kwargs)

    def _has_lm_head(self):
        # check module names of all modules inside `pretrained_model` to find the language model head
        for name, _module in self.pretrained_model.named_modules():
            if any(attribute in name for attribute in self.lm_head_namings):
                return True
        return False

    def post_init(self, state_dict):
        r"""
        We add the state dictionary of the value head to the state dictionary of the wrapped model by prepending the
        key with `v_head.`. This function removes the `v_head.` prefix from the keys of the value head state
        dictionary.
        """
        for k in list(state_dict.keys()):
            if "v_head." in k:
                state_dict[k.replace("v_head.", "")] = state_dict.pop(k)
        self.v_head.load_state_dict(state_dict, strict=False)
        del state_dict

        if hasattr(self.pretrained_model, "hf_device_map"):
            if (
                "cpu" in self.pretrained_model.hf_device_map.values()
                or "disk" in self.pretrained_model.hf_device_map.values()
            ):
                raise ValueError(
                    "The model is offloaded on CPU or disk - CPU & disk offloading is not supported for ValueHead models."
                )

            # get the lm_head device
            for name, module in self.pretrained_model.named_modules():
                if any(attribute in name for attribute in self.lm_head_namings):
                    lm_head_device = module.weight.device
                    break

            # put v_head on the same device as the lm_head to avoid issues
            self.v_head = self.v_head.to(lm_head_device)

            def set_device_hook(module, input, outputs):
                r"""
                A hook that sets the device of the output of the model to the device of the first parameter of the
                model.

                Args:
                    module (`nn.Module`):
                        The module to which the hook is attached.
                    input (`tuple`):
                        The input to the module.
                    outputs (`tuple`):
                        The output of the module.
                """
                new_output = ()
                for output in outputs:
                    if isinstance(output, torch.Tensor):
                        new_output += (output.to(lm_head_device),)
                    else:
                        new_output += (output,)
                return new_output

            self.register_forward_hook(set_device_hook)
            self.is_sequential_parallel = True

    def state_dict(self, *args, **kwargs):
        r"""
        Returns the state dictionary of the model. We add the state dictionary of the value head to the state
        dictionary of the wrapped model by prepending the key with `v_head.`.
        """
        if not self.is_peft_model:
            pretrained_model_state_dict = self.pretrained_model.state_dict(*args, **kwargs)
        else:
            # if it is a peft model, only save the v_head
            pretrained_model_state_dict = {}

        v_head_state_dict = self.v_head.state_dict(*args, **kwargs)
        for k, v in v_head_state_dict.items():
            pretrained_model_state_dict[f"v_head.{k}"] = v
        return pretrained_model_state_dict

    def push_to_hub(self, *args, **kwargs):
        self.pretrained_model.v_head = self.v_head

        return self.pretrained_model.push_to_hub(*args, **kwargs)

    def _init_weights(self, **kwargs):
        r"""
        We initialize the weights of the value head.
        """
        initializer_range = kwargs.pop("v_head_initializer_range", 0.2)
        # random init by default
        init_strategy = kwargs.pop("v_head_init_strategy", None)
        if init_strategy is None:
            # do nothing
            pass
        elif init_strategy == "normal":
            self.v_head.summary.weight.data.normal_(mean=0.0, std=initializer_range)
            self.v_head.summary.bias.data.zero_()

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        return_past_key_values=False,
        **kwargs,
    ):
        kwargs["past_key_values"] = past_key_values
        if self.is_peft_model and self.pretrained_model.active_peft_config.peft_type == "PREFIX_TUNING":
            kwargs.pop("past_key_values")

        base_model_output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,  # We force the model to output hidden states
            **kwargs,
        )

        last_hidden_state = base_model_output.decoder_hidden_states[-1]
        lm_logits = base_model_output.logits
        loss = base_model_output.loss

        value = self.v_head(last_hidden_state).squeeze(-1)

        # force upcast in fp32 if logits are in half-precision
        if lm_logits.dtype != torch.float32:
            lm_logits = lm_logits.float()

        if return_past_key_values:
            return (lm_logits, loss, value, base_model_output.past_key_values)
        else:
            return (lm_logits, loss, value)

    def generate(self, *args, **kwargs):
        r"""
        We call `generate` on the wrapped model.
        """
        return self.pretrained_model.generate(*args, **kwargs)
