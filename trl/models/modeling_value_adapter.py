import torch
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from torch import nn
from transformers import AutoModelForCausalLM

from trl import AutoModelForCausalLMWithValueHead


class AutoModelForCausalLMWithValueAdapter(AutoModelForCausalLMWithValueHead):
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
    transformers_parent_class = AutoModelForCausalLM
    lm_head_namings = ["lm_head", "embed_out"]
    supported_args = (
        "summary_dropout_prob",
        "v_head_initializer_range",
        "v_head_init_strategy",
    )

    def __init__(
        self, pretrained_model, policy_adapter_name="default", value_adapter_name="value_model_adapter", **kwargs
    ):
        super().__init__(pretrained_model, **kwargs)

        # simple v head to match reward model
        if hasattr(pretrained_model.config, "word_embed_proj_dim"):
            hidden_size = pretrained_model.config.word_embed_proj_dim
        else:
            hidden_size = pretrained_model.config.hidden_size
        self.v_head = nn.Linear(hidden_size, 1, bias=False)

        assert self.supports_rm_adapter, "value head adapter requires reward self adapter"
        self.policy_adapter_name = policy_adapter_name
        self.copy_reward_model_adapter_to_value_function(self.rm_adapter_name, value_adapter_name)
        self.value_adapter_name = value_adapter_name

    def copy_reward_model_adapter_to_value_function(
        self, reward_adapter_name="reward_model_adapter", value_adapter_name="value_function_adapter"
    ):
        # copy.deepcopy doesn't work with LoraConfig for some reason, so use this
        adapter_peft_config = LoraConfig(**self.pretrained_model.peft_config[reward_adapter_name].to_dict())
        adapter_peft_config.inference_mode = False
        self.pretrained_model.add_adapter(value_adapter_name, adapter_peft_config)

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

        self.pretrained_model.set_adapter(self.policy_adapter_name)
        self.pretrained_model.train()

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
        self.pretrained_model.train()

        value_model_output = self.pretrained_model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            **kwargs,
        )

        last_hidden_state = value_model_output.hidden_states[-1]

        if last_hidden_state.device != self.v_head.weight.device:
            last_hidden_state = last_hidden_state.to(self.v_head.weight.device)

        value = self.v_head(last_hidden_state).squeeze(-1)

        self.pretrained_model.set_adapter(self.policy_adapter_name)
        self.pretrained_model.train()

        # force upcast in fp32 if logits are in half-precision
        if lm_logits.dtype != torch.float32:
            lm_logits = lm_logits.float()

        return (lm_logits, loss, value)
