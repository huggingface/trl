# import torch
# from torch import nn
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

from .modeling_base import PreTrainedModelWrapper


class AutoModelForCausalLMWithValueModel(PreTrainedModelWrapper):
    transformers_parent_class = AutoModelForCausalLM

    def __init__(self, pretrained_model=None, value_model=None, **kwargs):
        super().__init__(pretrained_model)
        self.value_model = value_model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, value_model_name_or_path, *model_args, **kwargs):
        policy_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        value_model = AutoModelForSequenceClassification.from_pretrained(
            value_model_name_or_path, *model_args, **kwargs
        )

        model = cls(policy_model, value_model)

        model.is_peft_model = False
        # model.current_device = current_device

        return model

    def generate(self, *args, **kwargs):
        r"""
        A simple wrapper around the `generate` method of the wrapped model.
        Please refer to the [`generate`](https://huggingface.co/docs/transformers/internal/generation_utils)
        method of the wrapped model for more information about the supported arguments.

        Args:
            *args (`list`, *optional*):
                Positional arguments passed to the `generate` method of the wrapped model.
            **kwargs (`dict`, *optional*):
                Keyword arguments passed to the `generate` method of the wrapped model.
        """
        return self.pretrained_model.generate(*args, **kwargs)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        use_score=False,
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
        kwargs["past_key_values"] = past_key_values
        kwargs["output_hidden_states"] = True  # this had already been set in the LORA / PEFT examples

        if self.is_peft_model and self.pretrained_model.active_peft_config.peft_type == "PREFIX_TUNING":
            kwargs.pop("past_key_values")

        policy_model_output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        lm_logits = policy_model_output.logits
        loss = policy_model_output.loss

        value_model_output = self.value_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        last_hidden_state = value_model_output.hidden_states[-1]
        value = self.value_model.score(last_hidden_state).squeeze(-1)

        return (lm_logits, loss, value)
