from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from accelerate import PartialState
from datasets import builder, load_dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    PretrainedConfig,
    PreTrainedModel,
    Trainer,
)
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from trl import ModelConfig, RewardConfig


builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True

tqdm.pandas()


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with RewardTrainer
    """

    dataset_name: Optional[str] = field(
        default="mnoukhov/dpo_20konly_1b_fp16.yml_1a838_generations", metadata={"help": "the dataset name"}
    )
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "the tokenizer name"})


def get_kbit_device_map() -> Optional[Dict[str, int]]:
    # if is_xpu_available():
    #     return {"": f"xpu:{PartialState().local_process_index}"}
    if torch.cuda.is_available():
        return {"": PartialState().local_process_index}
    else:
        return None


def get_quantization_config(model_config: ModelConfig) -> Optional[BitsAndBytesConfig]:
    if model_config.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=model_config.torch_dtype,  # For consistency with model weights, we use the same value as `torch_dtype`
            bnb_4bit_quant_type=model_config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_config.use_bnb_nested_quant,
        )
    elif model_config.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None

    return quantization_config


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.normal_(layer.weight, std=std)
    torch.nn.init.constant_(layer.bias, val=bias_const)
    return layer


class ScalarModelConfig(PretrainedConfig):
    def __init__(
        self,
        base_model: str = "EleutherAI/pythia-160m",
        base_config: PretrainedConfig = AutoConfig.from_pretrained("EleutherAI/pythia-160m"),
        hidden_size: int = 768,
        bias: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.base_config = base_config
        self.hidden_size = hidden_size
        self.bias = bias


class ScalarModel(PreTrainedModel):
    config_class = ScalarModelConfig
    _supports_flash_attn_2 = True

    def __init__(self, config: ScalarModelConfig, **base_model_kwargs):
        super().__init__(config)
        # if config.base_model == "models/EleutherAI/pythia-6.9b-deduped/sft_model_55513":
        #     config.base_model = "vwxyzjn/EleutherAI_pythia-6.9b-deduped__sft__tldr"
        #     config.base_model_revision = "sft__55513__1706646024"
        #     config.base_config["_name_or_path"] = "vwxyzjn/EleutherAI_pythia-6.9b-deduped__sft__tldr"
        #     config.base_config["revision"] = "sft__55513__1706646024"
        self.config = config
        self.lm_backbone = AutoModel.from_pretrained(
            config.base_model,
            revision=getattr(config, "base_model_revision", None),
            config=self.config.base_config,
            trust_remote_code=True,
            **base_model_kwargs,
        )
        self.scalar_head = layer_init(
            nn.Linear(self.config.hidden_size, 1),
            std=1 / np.sqrt(self.config.hidden_size + 1),
        )

    def forward(self, input_ids, attention_mask, output_hidden_states=True, return_dict=True, **kwargs):
        output = self.lm_backbone(input_ids, attention_mask, output_hidden_states=True, return_dict=True, **kwargs)
        reward_logits = self.scalar_head(output.hidden_states[-1]) - self.config.bias

        sequence_lengths = first_true_indices(input_ids[:, :] == self.config.pad_token_id) - 1
        # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454
        reward = reward_logits[
            torch.arange(reward_logits.size(0), device=reward_logits.device), sequence_lengths
        ].squeeze(-1)

        return_dict = kwargs.pop("return_dict", None)
        if not return_dict:
            return_values = (reward,) + output[1:]
            # assume loss is None
            return return_values

        return SequenceClassifierOutputWithPast(
            loss=None,
            logits=reward,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )


def first_true_indices(bools, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values


if __name__ == "__main__":
    parser = HfArgumentParser((RewardConfig, ModelConfig, ScriptArguments))
    reward_config, model_config, script_args = parser.parse_args_into_dataclasses()
    reward_config.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    scalar_model_config = ScalarModelConfig.from_pretrained(
        model_config.model_name_or_path,
        revision=model_config.model_revision,
    )
    # hack to remove the path
    # models/EleutherAI/pythia-6.9b-deduped/sft_model_55513 -> EleutherAI/pythia-6.9b-deduped
    if scalar_model_config.base_model.startswith("models/"):
        original_model = scalar_model_config.base_config["_name_or_path"].split("/")[2]
        sft_model = f"vwxyzjn/EleutherAI_{original_model}__sft__tldr"
        scalar_model_config.base_config["_name_or_path"] = sft_model
        scalar_model_config.base_model = sft_model
        scalar_model_config.base_model_revision = "sft__55513__1706646024"

    quantization_config = get_quantization_config(model_config)
    model = ScalarModel.from_pretrained(
        model_config.model_name_or_path,
        revision=model_config.model_revision,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        config=scalar_model_config,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=reward_config,
    )

    datasets = []
    epochs = [1] + list(range(1000, 10000, 1000))
    for epoch in epochs:
        dataset = load_dataset(script_args.dataset_name, revision=str(epoch), split="train[:100]")

        dataset = dataset.map(
            lambda example: tokenizer(
                example["generations"] + "<|endoftext|>",
                padding="max_length",
                max_length=reward_config.max_length,
                truncation=True,
            ),
            batched=True,
        )

        results = trainer.predict(dataset)
        # raw_datasets = raw_datasets.filter(
        #     lambda x: len(x["input_ids_chosen"]) <= reward_config.max_length
        #     and len(x["input_ids_rejected"]) <= reward_config.max_length
        # )
        # train_dataset = raw_datasets["train"]
        # eval_dataset = raw_datasets["test"]
