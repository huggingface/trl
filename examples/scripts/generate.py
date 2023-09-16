from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, HfArgumentParser, AutoTokenizer, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
import torch
import os

@dataclass
class ScriptArguments:
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "the model name"})
    is_encoder_decoder: Optional[bool] = field(default=False, metadata={"help": "Is your model an encoder-decoder."})
    dataset_name: Optional[str] = field(default="Anthropic/hh-rlhf", metadata={"help": "the HF data path"})
    save_dataset_path: Optional[str] = field(default=None, metadata={"help": "the HF data path"})
    prompt_column_name: Optional[str] = field(default="prompt")
    generation_column_name: Optional[str] = field(default="generated")
    
    bs: Optional[int] = field(default=4, metadata={"help": "the generation batch size"})
    max_prompt_length: Optional[int] = field(default=256, metadata={"help": "The maximum prompt length"})
    
    truncation_side: Optional[int] = field(default="right", metadata={"help": "the side to truncate the prompt if the prompt is longer than max_prompt_length"})
    max_new_tokens: Optional[int] = field(default=256, metadata={"help": "the maximum number of tokens generated per sample"})
    temperature: Optional[float] = field(default=1.)
    top_p: Optional[float] = field(default=1.)
    top_k: Optional[float] = field(default=50)
    num_return_sequences: Optional[int] = field(default=1)
    size: Optional[float] = field(default=1., metadata={"help": "Percentage of the dataset you want to make generation on."})
    bf16: Optional[bool] = field(default=True if torch.cuda.get_device_capability()[0] == 8 else False, metadata={"help": "whether to use bf16."})
    fp16: Optional[bool] = field(default=True if not torch.cuda.get_device_capability()[0] == 8 else False, metadata={"help": "whether to use fp16."})

def main():

    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    accelerator = Accelerator(
        mixed_precision= "bf16" if script_args.bf16 else "fp16" if script_args.fp16 else "no"
    )
    
    gen_kwargs = {
        "max_new_tokens": script_args.max_new_tokens,
        "temperature": script_args.temperature,
        "num_return_sequences": script_args.num_return_sequences,
        "do_sample": True,
        "top_p": script_args.top_p,
        "top_k": script_args.top_k
    }
    
    
    # Load model, reward model and tokenizer
    if script_args.is_encoder_decoder:
        model = AutoModelForSeq2SeqLM.from_pretrained(script_args.model_name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path)
        
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    
    # Load and extract the prompt in the dataset. We do not need this step if we already have a dataset with prompts separated from the answers.
    gen_dataset = load_dataset(script_args.dataset_name, split="train")
    
    if script_args.size!=1.:
        gen_dataset_length = len(gen_dataset)
        gen_dataset = gen_dataset.select([i for i in range(int(len(gen_dataset)*script_args.size))])
        print(f"Decreasing the size of the generation dataset from {gen_dataset_length} to {len(gen_dataset)}.")


    def preprocess_function(sample):
        def extract_prompt(chosen, rejected):
            for i, (c, r) in enumerate(zip(chosen, rejected)):
                if c != r:
                    return chosen[:i].strip()
            return chosen

        prompts = []
        for chosen, rejected in zip(sample["chosen"], sample["rejected"]):
            prompts.append(extract_prompt(chosen, rejected))
        model_inputs = tokenizer(prompts, max_length=512, truncation=True)

        return model_inputs

    gen_dataset = gen_dataset.map(preprocess_function, batched=True, remove_columns=list(gen_dataset.features))
    
    
    if script_args.is_encoder_decoder:
        collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)
    else:
        tokenizer.padding_side = "left"
        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8)
        
    gen_dataloader = DataLoader(gen_dataset, batch_size=script_args.bs, shuffle=False, collate_fn=collator)
    
    model, gen_dataloader = accelerator.prepare(model, gen_dataloader)
    
    accelerator.wait_for_everyone()
    
    all_predictions = []
    all_prompts = []
    
    for batch in gen_dataloader:
        with torch.no_grad():
            prompt_len = batch["input_ids"].shape[1]
            
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )
            
            if not script_args.is_encoder_decoder:
                generated_tokens = generated_tokens[:, prompt_len:]
                
            prompts = batch["input_ids"].repeat_interleave(script_args.num_return_sequences, dim=0)

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            
            prompts = accelerator.pad_across_processes(
                prompts, dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens = accelerator.gather(generated_tokens)
            generated_tokens = generated_tokens.cpu().numpy()
            
            prompts = accelerator.gather(prompts)
            prompts = prompts.cpu().numpy()

            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]

            all_predictions.extend(generated_tokens)
            all_prompts.extend(prompts)

    all_prompts = tokenizer.batch_decode(all_prompts, skip_special_tokens=True)
    all_predictions = tokenizer.batch_decode(all_predictions, skip_special_tokens=True)
    
    generated_dataset = Dataset.from_dict(
        {
            script_args.prompt_column_name: all_prompts,
            script_args.generation_column_name: all_predictions
        }
    )
    
    # Don't forget to concatenate the two datasets
    # Make it only for decoders only models so that we can use the sft trainer
    
    generated_dataset.save_to_disk(os.path.join(script_args.dataset_path), "train")
            
if __name__=="__main__":
    main()