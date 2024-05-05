from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from accelerate import PartialState
from datasets import builder, load_dataset
from gpt_reward_modeling import GPTRewardDataCollatorWithPadding, GPTRewardTrainer
from torch.utils.data import DataLoader
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
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutputWithPast


# from trl import ModelConfig, RewardConfig


builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True

tqdm.pandas()


@dataclass
class ScriptArguments:
    dataset_name: Optional[str] = field(default=None)
    dataset_split: Optional[str] = field(default=None)
    wandb_log_id: Optional[str] = field(default=None)
    model_name: Optional[str] = field(default="EleutherAI/pythia-410m", metadata={"help": "the model name"})
    model_revision: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "the tokenizer name"})
    eval_dtype: Optional[str] = field(default="auto")
    eval_batch_size: Optional[int] = field(default=16)
    max_length: Optional[int] = field(default=512)
    flash_attention: Optional[bool] = field(default=False)


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


def relabel_with_preds(batch: Dict[str, List]):
    relabel_batch = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
        "pred_chosen": [],
        "pred_rejected": [],
    }
    for prompt, chosen, rejected, pred_chosen, pred_rejected in zip(
        batch["prompt"],
        batch["chosen"],
        batch["rejected"],
        batch["pred_chosen"],
        batch["pred_rejected"],
    ):
        relabel_batch["prompt"].append(prompt)
        if pred_chosen >= pred_rejected:
            relabel_batch["chosen"].append(chosen)
            relabel_batch["rejected"].append(rejected)
            relabel_batch["pred_chosen"].append(pred_chosen)
            relabel_batch["pred_rejected"].append(pred_rejected)
        else:
            relabel_batch["chosen"].append(rejected)
            relabel_batch["rejected"].append(chosen)
            relabel_batch["pred_chosen"].append(pred_rejected)
            relabel_batch["pred_rejected"].append(pred_chosen)

    return relabel_batch


def prepare_dataset(args, dataset, tokenizer):
    original_columns = dataset.column_names

    def preprocess_function(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
            full_chosen = prompt + chosen
            if not full_chosen.endswith(tokenizer.eos_token):
                full_chosen += tokenizer.eos_token

            full_rejected = prompt + rejected
            if not full_rejected.endswith(tokenizer.eos_token):
                full_rejected += tokenizer.eos_token

            chosen_tokenized = tokenizer(
                full_chosen,
                padding="max_length",
                max_length=args.max_length,
                truncation=True,
            )

            rejected_tokenized = tokenizer(
                full_rejected,
                padding="max_length",
                max_length=args.max_length,
                truncation=True,
            )

            # guarantee that last token is EOS if truncated
            token_length = sum(chosen_tokenized["attention_mask"])
            if token_length == args.max_length:
                chosen_tokenized["input_ids"][-1] = tokenizer.eos_token_id

            token_length = sum(rejected_tokenized["attention_mask"])
            if token_length == args.max_length:
                rejected_tokenized["input_ids"][-1] = tokenizer.eos_token_id

            new_examples["input_ids_chosen"].append(chosen_tokenized["input_ids"])
            new_examples["attention_mask_chosen"].append(chosen_tokenized["attention_mask"])
            new_examples["input_ids_rejected"].append(rejected_tokenized["input_ids"])
            new_examples["attention_mask_rejected"].append(rejected_tokenized["attention_mask"])

        return new_examples

    dataset = dataset.map(preprocess_function, batched=True, remove_columns=original_columns)

    return dataset


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    torch_dtype = args.eval_dtype if args.eval_dtype in ["auto", None] else getattr(torch, args.eval_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    scalar_model_config = ScalarModelConfig.from_pretrained(
        args.model_name,
        revision=args.model_revision,
    )
    # hack to remove the path
    # models/EleutherAI/pythia-6.9b-deduped/sft_model_55513 -> EleutherAI/pythia-6.9b-deduped
    if scalar_model_config.base_model.startswith("models/"):
        original_model = scalar_model_config.base_config["_name_or_path"].split("/")[2]
        sft_model = f"vwxyzjn/EleutherAI_{original_model}__sft__tldr"
        scalar_model_config.base_config["_name_or_path"] = sft_model
        scalar_model_config.base_model = sft_model
        _, seed, _ = args.model_revision.split("__")
        scalar_model_config.base_model_revision = f"sft__{seed}__1708611267"

    # quantization_config = get_quantization_config(model_config)
    model = ScalarModel.from_pretrained(
        args.model_name,
        revision=args.model_revision,
        config=scalar_model_config,
        torch_dtype=torch_dtype,
        use_flash_attention_2=args.flash_attention,
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    ## get reference continuation rewards
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    dataset = prepare_dataset(args, dataset, tokenizer)

    data_collator = GPTRewardDataCollatorWithPadding(tokenizer, max_length=args.max_length)

    training_args = TrainingArguments(per_device_eval_batch_size=int(args.eval_batch_size), output_dir=".")

    trainer = GPTRewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        max_length=args.max_length,
        data_collator=data_collator,
    )

    outputs = trainer.predict(dataset)
    preds = outputs.predictions

    import pdb

    pdb.set_trace()
