# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    HfArgumentParser, 
    TrainingArguments, 
    AutoTokenizer
)
from trl import (
    SFTTrainer, 
    DataCollatorForCompletionOnlyLM
)

tqdm.pandas()


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """
    model_name_or_path: Optional[str] = field(
        default="facebook/opt-350m", metadata={"help": "the model name"}
    )
    train_file: Optional[str] = field(
        default="timdettmers/openassistant-guanaco", metadata={"help": "the dataset name"}
    )
    dataset_text_field: Optional[str] = field(
        default="text", metadata={"help": "the text field of the dataset"}
    )
    log_with: Optional[str] = field(
        default=None, metadata={"help": "use 'wandb' to log with wandb"}
    )
    learning_rate: Optional[float] = field(
        default=1.41e-5, metadata={"help": "the learning rate"}
    )
    per_device_train_batch_size: Optional[int] = field(
        default=64, metadata={"help": "the batch size"}
    )
    seq_length: Optional[int] = field(
        default=512, metadata={"help": "Input sequence length"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    torch_dtype: Optional[str] = field(
        default="float16", metadata={"help": "the dtype of the model"}
    )
    load_in_8bit: Optional[bool] = field(
        default=False, metadata={"help": "load the model in 8 bits precision"}
    )
    load_in_4bit: Optional[bool] = field(
        default=False, metadata={"help": "load the model in 4 bits precision"}
    )
    use_peft: Optional[bool] = field(
        default=False, metadata={"help": "Wether to use PEFT or not to train adapters"}
    )
    trust_remote_code: Optional[bool] = field(
        default=True, metadata={"help": "Enable `trust_remote_code`"}
    )
    output_dir: Optional[str] = field(
        default="output", metadata={"help": "the output directory"}
    )
    peft_lora_r: Optional[int] = field(
        default=64, metadata={"help": "the r parameter of the LoRA adapters"}
    )
    peft_lora_alpha: Optional[int] = field(
        default=16, metadata={"help": "the alpha parameter of the LoRA adapters"}
    )
    logging_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of logging steps"}
    )
    use_auth_token: Optional[bool] = field(
        default=True, metadata={"help": "Use HF auth token to access the model"}
    )
    num_train_epochs: Optional[int] = field(
        default=3, metadata={"help": "the number of training epochs"}
    )
    max_steps: Optional[int] = field(
        default=-1, metadata={"help": "the number of training steps"}
    )
    save_steps: Optional[int] = field(
        default=100, metadata={"help": "Number of updates steps before two checkpoint saves"}
    )
    save_total_limit: Optional[int] = field(
        default=10, metadata={"help": "Limits total number of checkpoints."}
    )
    push_to_hub: Optional[bool] = field(
        default=False, metadata={"help": "Push the model to HF Hub"}
    )
    hub_model_id: Optional[str] = field(
        default=None, metadata={"help": "The name of the model on HF Hub"}
    )
    deepspeed: Optional[str] = field(
        default=None, metadata={"help": "DeepSpeed training configuration file"}
    )
    validatoin_split_percentage: Optional[int] = field(
        default=5, metadata={"help": "The percentage of the train set used as validation set"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "The cache directory"}
    )
    use_fast_tokenizer: Optional[bool] = field(
        default=True, metadata={"help": "Use fast tokenizer"}
    )
    model_revision: Optional[str] = field(
        default="main", metadata={"help": "The revision of the model"}
    )
    use_auth_token: Optional[bool] = field(
        default=False, metadata={"help": "Use HF auth token to access the model"}
    )
    bf16: Optional[bool] = field(
        default=False, metadata={"help": "Use bfloat16 precision"}
    )
    fp16: Optional[bool] = field(
        default=False, metadata={"help": "Use fp16 precision"}
    )
    instruction_template: Optional[str] = field(
        default=None, metadata={"help": "instruction_template"}
    )
    response_template: Optional[str] = field(
        default=None, metadata={"help": "response_template"}
    )

# main().
def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Step 1: Load the model
    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif script_args.load_in_8bit or script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
        )
        # Copy the model to each device
        device_map = {"": Accelerator().local_process_index}
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        
        # torch_dtype
        if script_args.torch_dtype == "float16":
            torch_dtype = torch.float16
        elif script_args.torch_dtype == "float32":
            torch_dtype = torch.float32
        elif script_args.torch_dtype == "bfloat16":
            # for Amphere GPU (A100, A6000...)
            # if you have trouble which lr doesn't decrease, try to use bfloat16
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = None

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code,
        torch_dtype=torch_dtype,
        use_auth_token=script_args.use_auth_token,
    )
    
    # Step 2: Load the tokenizer
    tokenizer_kwargs = {
        "cache_dir": script_args.cache_dir,
        "use_fast": script_args.use_fast_tokenizer,
        "revision": script_args.model_revision,
        "use_auth_token": True if script_args.use_auth_token else None,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        **tokenizer_kwargs,
        )

    # Step 3: Load the dataset
    # Currently, you can only use jsonl file.
    # If you want to use other file, you should modify this code.
    if script_args.train_file.endswith(".jsonl"):
        dataset = load_dataset(
            "json",
            data_files=script_args.train_file,
            split='train',
            )
    else:
        raise ValueError("You should use jsonl.")

    # Step 4: DataCollator
    
    # Step 4-1. formatting_prompts_func
    """
    It's up to your dataset format. 
    If you want to modify, then check the trl document.
    https://huggingface.co/docs/trl/sft_trainer
    """
    def formatting_prompts_func(example):
        return example[script_args.dataset_text_field]
        
    # Step 4-2. data_collator
    # You should set the response_template.
    instruction_template = script_args.instruction_template
    response_template = script_args.response_template
    
    # Only use response_template
    if instruction_template is None and response_template is not None:
        collator = DataCollatorForCompletionOnlyLM(
            response_template, 
            tokenizer=tokenizer
            )
    # Use instruction_template and response_template both for assistant style conversation data
    elif instruction_template is not None and response_template is not None:
        """
        To instantiate that collator for assistant style conversation data, 
        pass a response template, an instruction template and the tokenizer. 
        to fine-tune llm on assistant completions (response_template) only
        """
        collator = DataCollatorForCompletionOnlyLM(
            instruction_template, 
            response_template, 
            tokenizer=tokenizer
            )
    else:
        raise ValueError("You should use instruction_template and response_template.")

    # Step 5: Define the training arguments
    training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        logging_steps=script_args.logging_steps,
        num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.max_steps,
        report_to=script_args.log_with,
        save_steps=script_args.save_steps,
        save_total_limit=script_args.save_total_limit,
        push_to_hub=script_args.push_to_hub,
        hub_model_id=script_args.hub_model_id,
        deepspeed=script_args.deepspeed, # deepspeed option
        bf16=script_args.bf16, # bf16 mixed-precision
        fp16=script_args.fp16 # fp16 mixed-precision
    )

    # Step 6: Define the LoraConfig (Optional)
    if script_args.use_peft:
        peft_config = LoraConfig(
            r=script_args.peft_lora_r,
            lora_alpha=script_args.peft_lora_alpha,
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None


    # Step 7: Define the Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        max_seq_length=script_args.seq_length,
        train_dataset=dataset,
        dataset_text_field=script_args.dataset_text_field,
        formatting_func=formatting_prompts_func, # formatting_prompts_func
        data_collator=collator, # data_collator
        peft_config=peft_config, # peft option
    )
    trainer.train()

    # Step 8: Save the model
    trainer.save_model(script_args.output_dir)


if __name__ == "__main__":
    main()