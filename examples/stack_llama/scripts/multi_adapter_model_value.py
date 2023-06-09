import copy

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from trl import AutoModelForCausalLMWithValueHead


class AutoModelForCausalLMWithMultiAdapterValueHead(AutoModelForCausalLMWithValueHead):
    r"""
    An extension of AutoModelForCausalLM that adds an adapter for the value function if
    there is an adapter for the reward model. In this way, there are no shared parameters
    between the policy and the value function, and the value function is initialized from
    the reward model as recommended by Ouyang et al (2020)

    Class attributes:
        - **transformers_parent_class** (`transformers.PreTrainedModel`) -- The parent class of the wrapped model. This
            should be set to `transformers.AutoModelForCausalLM` for this class.
        - **lm_head_namings** (`tuple`) -- A tuple of strings that are used to identify the language model head of the
            wrapped model. This is set to `("lm_head", "embed_out")` for this class but can be changed for other models
            in the future
        - **supported_args** (`tuple`) -- A tuple of strings that are used to identify the arguments that are supported
            by the `ValueHead` class. Currently, the supported args are:
            - **summary_dropout_prob** (`float`, `optional`, defaults to `None`) -- The dropout probability for the
                `ValueHead` class.
            - **v_head_initializer_range** (`float`, `optional`, defaults to `0.2`) -- The initializer range for the
                `ValueHead` if a specific initialization strategy is selected.
            - **v_head_init_strategy** (`str`, `optional`, defaults to `None`) -- The initialization strategy for the
                `ValueHead`. Currently, the supported strategies are:
                - **`None`** -- Initializes the weights of the `ValueHead` with a random distribution. This is the default
                    strategy.
                - **"normal"** -- Initializes the weights of the `ValueHead` with a normal distribution.

    """
    transformers_parent_class = AutoModelForCausalLMWithValueHead
    lm_head_namings = ["lm_head", "embed_out"]
    supported_args = (
        "summary_dropout_prob",
        "v_head_initializer_range",
        "v_head_init_strategy",
    )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Instantiates a new model from a pretrained model from `transformers`. The
        pretrained model is loaded using the `from_pretrained` method of the
        `transformers.PreTrainedModel` class. The arguments that are specific to the
        `transformers.PreTrainedModel` class are passed along this method and filtered
        out from the `kwargs` argument.


        Args:
            pretrained_model_name_or_path (`str` or `transformers.PreTrainedModel`):
                The path to the pretrained model or its name.
            *model_args (`list`, *optional*)):
                Additional positional arguments passed along to the underlying model's
                `from_pretrained` method.
            **kwargs (`dict`, *optional*):
                Additional keyword arguments passed along to the underlying model's
                `from_pretrained` method. We also pre-process the kwargs to extract
                the arguments that are specific to the `transformers.PreTrainedModel`
                class and the arguments that are specific to trl models. The kwargs
                also support `prepare_model_for_int8_training` arguments from
                `peft` library.
        """
        if kwargs is not None:
            value_adapter_name = kwargs.pop("value_adapter_name", "value_model_adapter")
            policy_adapter_name = kwargs.get("adapter_name", "default")
        else:
            value_adapter_name = "value_model_adapter"
            policy_adapter_name = "default"

        model = cls.transformers_parent_class.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        model.policy_adapter_name = policy_adapter_name

        # if reward_adapter is not None and not isinstance(reward_adapter, str):
        #     raise ValueError(
        #         "The `reward_adapter` argument should be a string representing the name of local path or the Hub id to the Reward Modeling adapter."
        #     )
        #

        if model.supports_rm_adapter:
            model.copy_reward_modeling_adapter_to_value_function(model.rm_adapter_name, value_adapter_name)

        return model

    def copy_reward_modeling_adapter_to_value_function(self, reward_adapter_name="reward_model_adapter", value_adapter_name="value_function_adapter"):
        adapter_peft_config = copy.deepcopy(self.pretrained_model.peft_config[reward_adapter_name])
        self.pretrained_model.add_adapter(value_adapter_name, adapter_peft_config)

        self.value_adapter_name = value_adapter_name

        # load the reward model's adapter and head state to the value function
        self.v_head.load_state_dict(self.score.state_dict())

        adapter_state_dict = get_peft_model_state_dict(self.pretrained_model, adapter_name=reward_adapter_name)
        set_peft_model_state_dict(self.pretrained_model, adapter_state_dict, adapter_name=value_adapter_name)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
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
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the wrapped model.
        """
        kwargs["output_hidden_states"] = True  # this had already been set in the LORA / PEFT examples

        base_model_output = self.pretrained_model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            **kwargs,
        )

        lm_logits = base_model_output.logits
        loss = base_model_output.loss

        # value model
        self.pretrained_model.set_adapter(self.value_adapter_name)

        value_model_output = self.pretrained_model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            **kwargs,
        )

        last_hidden_state = value_model_output.hidden_states[-1]

        if last_hidden_state.device != self.v_head.summary.weight.device:
            last_hidden_state = last_hidden_state.to(self.v_head.summary.weight.device)

        value = self.v_head(last_hidden_state).squeeze(-1)

        self.pretrained_model.set_adapter(self.policy_adapter_name)

        # force upcast in fp32 if logits are in half-precision
        if lm_logits.dtype != torch.float32:
            lm_logits = lm_logits.float()

        return (lm_logits, loss, value)
