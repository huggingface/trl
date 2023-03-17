from dataclasses import dataclass, field
from typing import Optional

import peft
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable
    model_name: Optional[str] = field(default="edbeeching/gpt-neo-125M-imdb-lora", metadata={"help": "the model name"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

peft_model_id = script_args.model_name
peft_config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    return_dict=True,
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)
model.eval()

key_list = [key for key, _ in model.base_model.model.named_modules() if "lora" not in key]
for key in key_list:
    parent, target, target_name = model.base_model._get_submodules(key)
    if isinstance(target, peft.tuners.lora.Linear):
        bias = target.bias is not None
        new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
        model.base_model._replace_module(parent, target_name, new_module, target)

model = model.base_model.model

model.push_to_hub(f"{script_args.model_name}-adapter-merged", use_temp_dir=False)
