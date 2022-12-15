import torch.nn as nn
from transformers import PreTrainedModel

class ValueHead(nn.Module):
    """The ValueHead class implements a head for GPT2 that returns a scalar for each output token."""
    def __init__(self, config):
        super().__init__()
        self.detach_head = False
        self.summary_type = config.summary_type if hasattr(config, "summary_type") else "last"
        if self.summary_type == "attn":
            raise NotImplementedError

        self.summary = nn.Identity()
        if hasattr(config, "summary_use_proj") and config.summary_use_proj:
            if hasattr(config, "summary_proj_to_labels") and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = nn.Linear(config.hidden_size, num_classes)

        self.activation = nn.Identity()
        if hasattr(config, "summary_activation") and config.summary_activation == "tanh":
            self.activation = nn.Tanh()

        self.first_dropout = nn.Identity()
        if hasattr(config, "summary_first_dropout") and config.summary_first_dropout > 0:
            self.first_dropout = nn.Dropout(config.summary_first_dropout)

        self.last_dropout = nn.Identity()
        if hasattr(config, "summary_last_dropout") and config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(config.summary_last_dropout)

        self.flatten = nn.Flatten()

    def forward(self, hidden_states, cls_index=None):
        if self.detach_head:
            output = hidden_states.detach()
        else:
            output = hidden_states
        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)

        return output


class AutoRegressiveLMWithValueHead(PreTrainedModel):
    def __init__(self, autoregressive_model):
        super().__init__(autoregressive_model.config)
        if not hasattr(autoregressive_model, "lm_head"):
            raise ValueError("The autoregressive model must have a language modeling head, please load a model using `AutoModelForCausalLM.from_pretrained(...)`")

        self.autoregressive_model = autoregressive_model
        self.v_head = ValueHead(self.config)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        base_model_output = self.autoregressive_model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )

        last_hidden_state = base_model_output.hidden_states[-1]
        lm_logits = base_model_output.logits
        loss = base_model_output.loss

        value = self.v_head(last_hidden_state).squeeze(-1)

        return (lm_logits, loss, value)
    
    def generate(
        self,
        input_ids,
        **generate_kwargs,
    ):
        r"""
        We call `generate` on the autoregressive model with the proper arguments.
        """
        return self.autoregressive_model.generate(input_ids, **generate_kwargs)
    
    def push_to_hub(self, *args, **kwargs):
        r"""
        Since we expect the `generate` method to be applied on the autoregressive model, we push the autoregressive model to the hub.
        The value head is not pushed to the hub.
        """
        return self.autoregressive_model.push_to_hub(*args, **kwargs)
    
    def save_pretrained(self, *args, **kwargs):
        r"""
        Since we expect the `generate` method to be applied on the autoregressive model, we save the autoregressive model.
        The value head is not saved.
        """
        return self.autoregressive_model.save_pretrained(*args, **kwargs)