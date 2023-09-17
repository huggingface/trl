# [WIP]
from transformers import AutoModelForCausalLM, HfArgumentParser, AutoTokenizer, DataCollatorForLanguageModeling
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
from datasets import Dataset, load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

@dataclass
class ScriptArguments:
    model_name_or_path: Optional[str] = field(default="gpt2", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(default="Anthropic/hh-rlhf", metadata={"help": "the HF data path"})
    
    save_dataset_path: Optional[str] = field(default=None, metadata={"help": "the save dataset path"})
    generation_column_name: Optional[str] = field(default="generated")
    
    gen_bs: Optional[int] = field(default=4, metadata={"help": "the generation batch size"})
    max_prompt_length: Optional[int] = field(default=256, metadata={"help": "The maximum prompt length"})
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "Percentage of the dataset you want to make generation on."})
    bf16: Optional[bool] = field(default=True if torch.cuda.get_device_capability()[0] == 8 else False, metadata={"help": "whether to use bf16."})
    fp16: Optional[bool] = field(default=True if not torch.cuda.get_device_capability()[0] == 8 else False, metadata={"help": "whether to use fp16."})
    
    max_new_tokens: Optional[int] = field(default=256, metadata={"help": "the maximum number of tokens generated per sample"})
    temperature: Optional[float] = field(default=1.)
    top_p: Optional[float] = field(default=1.)
    top_k: Optional[float] = field(default=50)
    num_return_sequences: Optional[int] = field(default=1)
    
    split: Optional[str] = field(default = "train")
    
def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = "\n\nAssistant:"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]
    
def generate(script_args, save_dataset_path = None):
    
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
    model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path)  
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    dataset = load_dataset(script_args.dataset_name, split=script_args.split)
    
    if script_args.sanity_check:
        dataset = dataset.select(range(min(len(dataset), 500)))

    def preprocess_function(sample):
        prompts = []
        for chosen in sample["chosen"]:
            prompts.append(extract_anthropic_prompt(chosen))
        model_inputs = tokenizer(prompts, max_length=script_args.max_prompt_length, truncation=True)

        return model_inputs

    dataset = dataset.map(preprocess_function, batched=True, remove_columns=list(dataset.features))
    
    tokenizer.padding_side = "left"
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8)
        
    dataloader = DataLoader(dataset, batch_size=script_args.gen_bs, shuffle=False, collate_fn=collator)
    
    model, dataloader = accelerator.prepare(model, dataloader)
    
    accelerator.wait_for_everyone()
    
    all_predictions = []
    all_prompts = []
    pbar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process)
    # to be verified
    for batch in dataloader:
        with torch.no_grad():
            
            sequence_length = batch["input_ids"].shape[1]
            
            all_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pad_token_id=tokenizer.pad_token_id,
                **gen_kwargs,
            )

            generated_tokens = torch.tensor([tokens[sequence_length:].tolist() for tokens in all_tokens], device=accelerator.device)
            prompt_tokens = torch.tensor([tokens[:sequence_length].tolist() for tokens in all_tokens], device=accelerator.device)
            
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            
            prompt_tokens = accelerator.pad_across_processes(
                prompt_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens = accelerator.gather(generated_tokens)
            generated_tokens = generated_tokens.cpu()
            prompt_tokens = accelerator.gather(prompt_tokens)
            prompt_tokens = prompt_tokens.cpu()

            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
                prompt_tokens = prompt_tokens[0]

            all_predictions.extend(generated_tokens)
            all_prompts.extend(prompt_tokens)
            pbar.update(1)

    accelerator.wait_for_everyone()
    
    all_predictions = tokenizer.batch_decode(all_predictions, skip_special_tokens=True)[:len(dataset)]
    
    # postprocessing
    all_predictions = [preds.split("Human:")[0].strip() for preds in all_predictions]
    all_prompts = tokenizer.batch_decode(all_prompts, skip_special_tokens=True)[:len(dataset)]
    
    generated = [prompt + " " + preds for prompt, preds in zip(all_prompts, all_predictions)]
    
    generated_dataset = Dataset.from_dict(
        {
            script_args.generation_column_name: generated
        }
    )
    
    accelerator.print(generated_dataset[0])
    
    # Concatenate the two datasets
    dataset = load_dataset(script_args.dataset_name, split="train")
    
    columns_to_remove = list(dataset.features)
    columns_to_remove.remove("chosen")
    dataset = dataset.rename_column("chosen", script_args.generation_column_name)
    dataset = dataset.remove_columns(columns_to_remove)
    
    d_g = concatenate_datasets([dataset, generated_dataset])
    
    if accelerator.is_local_main_process:
        if save_dataset_path is None:
            d_g.save_to_disk(script_args.save_dataset_path)
        else:
            d_g.save_to_disk(save_dataset_path)
    
    return d_g
    
def main():
    
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    generate(script_args)
            
if __name__=="__main__":
    main()