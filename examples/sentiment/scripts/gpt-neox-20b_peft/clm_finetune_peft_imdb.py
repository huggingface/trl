from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_int8_training
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default="facebook/opt-125m",
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default="imdb", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    block_size: Optional[int] = field(
        default=1024, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )


parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

model_args, data_args, training_args = parser.parse_args_into_dataclasses()


model = AutoModelForCausalLM.from_pretrained(
    model_args.model_name_or_path,
    load_in_8bit=True,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
# ### Prepare model for training
#
# Some pre-processing needs to be done before training such an int8 model using `peft`, therefore let's import an utiliy function `prepare_model_for_int8_training` that will:
# - Cast the layer norm in `float32` for stability purposes
# - Add a `forward_hook` to the input embedding layer to enable gradient computation of the input hidden states
# - Enable gradient checkpointing for more memory-efficient training
# - Cast the output logits in `float32` for smoother sampling during the sampling procedure


if "gpt-neox" in model_args.model_name_or_path:
    model = prepare_model_for_int8_training(
        model, output_embedding_layer_name="embed_out", layer_norm_names=["layer_norm", "layernorm"]
    )
else:
    model = prepare_model_for_int8_training(model)


# ### Apply LoRA
#
# Here comes the magic with `peft`! Let's load a `PeftModel` and specify that we are going to use low-rank adapters (LoRA) using `get_peft_model` utility function from `peft`.
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


target_modules = None
if "gpt-neox" in model_args.model_name_or_path:
    target_modules = ["query_key_value", "xxx"]  # workaround to use 8bit training on this model
config = LoraConfig(
    r=16, lora_alpha=32, target_modules=target_modules, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

block_size = data_args.block_size


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


# ### Training
data = load_dataset("imdb")
columns = data["train"].features
data = data.map(lambda samples: tokenizer(samples["text"]), batched=True, remove_columns=columns)
data = data.map(group_texts, batched=True)

model.gradient_checkpointing_enable()
trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

# ## Share adapters on the 🤗 Hub
model.push_to_hub(training_args.output_dir, use_auth_token=True)

# Load adapters from the Hub and generate some output texts:

peft_model_id = training_args.output_dir
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)
# You can then directly use the trained model or the model that you have loaded from the 🤗 Hub for inference

batch = tokenizer("I really enjoyed the ", return_tensors="pt")

with torch.cuda.amp.autocast():
    output_tokens = model.generate(**batch, max_new_tokens=50)

print("\n\n", tokenizer.decode(output_tokens[0], skip_special_tokens=True))
