import os
from dataclasses import dataclass, field
from typing import Optional, List

from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
import wandb
from trl.experimental.online_dpo import OnlineDPOConfig, OnlineDPOTrainer
from multi_reward_models import RewardModels
import torch
#callbacks
from trl import LogCompletionsCallback
from transformers import GenerationConfig


@dataclass
class ScriptArguments:
    # logging / output (ppo.py-style)
    disable_wandb: bool = field(default=False, metadata={"help": "Disable wandb"})
    wandb_name: str = field(default="online_dpo", metadata={"help": "Run name"})
    save_directory: str = field(default="../online_dpo", metadata={"help": "Root output directory"})

    # dataset (CSV with a prompt column)
    csv_path: str = field(
        default="../datasets/anthropic/anthropic_train_deduped.csv",
        metadata={"help": "Path to CSV file containing prompts"},
    )
    prompt_column: str = field(default="prompt", metadata={"help": "CSV column name for prompt"})

    # model selection (similar to ppo.py)
    exp_type: str = field(default="assistant", metadata={"help": "assistant or summary"})
    sft_model_path: Optional[str] = field(default=None, metadata={"help": "Override policy model path"})

    # reward models (names mapped to HF ids below)
    reward_models: List[str] = field(
        default_factory=lambda: ["helpful","harmless"],
        metadata={"help": "Reward model names to use (e.g., helpful, harmless, humor, summary, faithful, deberta)"},
    )

    # optional LoRA (handled by trainer via peft_config)
    use_lora: bool = field(default=True)
    lora_rank: int = field(default=64)


if __name__ == "__main__":
    accelerator = Accelerator()

    parser = HfArgumentParser((ScriptArguments, OnlineDPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()

    if script_args.disable_wandb:
        training_args.report_to = []
    else:
        training_args.report_to = ["wandb"]
   

    # Policy model path: mimic your ppo.py behavior
    dataset_name = "anthropic" if script_args.exp_type == "assistant" else "summary"
    if script_args.sft_model_path is not None:
        base_model_name = script_args.sft_model_path
    else:
        base_model_name = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", dataset_name))

    # Output dir: keep ppo.py style
    if training_args.output_dir is None or training_args.output_dir == "trainer_output":
        training_args.output_dir = os.path.join(script_args.save_directory, script_args.wandb_name)
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Reward model name -> HF path mapping (same as ppo.py)
    REWARD_MODEL_PATHS = {
        "summary": "Tristan/gpt2_reward_summarization",
        "faithful": "CogComp/bart-faithful-summary-detector",
        "helpful": "Ray2333/gpt2-large-helpful-reward_model",
        "harmless": "Ray2333/gpt2-large-harmless-reward_model",
        "humor": "mohameddhiab/humor-no-humor",
        "deberta": "OpenAssistant/reward-model-deberta-v3-large-v2",
    }
   
    reward_model_paths = []
    for name in script_args.reward_models:
        if name not in REWARD_MODEL_PATHS:
            raise ValueError(f"Unknown reward model '{name}'. Options: {list(REWARD_MODEL_PATHS.keys())}")
        reward_model_paths.append(REWARD_MODEL_PATHS[name])

    # Tokenizers (decoder-only => left padding)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Policy model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=(None if training_args.fp16 else getattr(__import__("torch"), "bfloat16", None))
        if training_args.bf16
        else None,
    )
    
    model.generation_config.max_new_tokens = 128 if script_args.exp_type == "assistant" else 48
    model.generation_config.temperature = 1.0
    model.generation_config.top_k = 15
    model.generation_config.top_p = 1.0
    model.generation_config.do_sample = True
    model.generation_config.begin_suppress_tokens = [tokenizer.eos_token_id]
    model.generation_config.max_length = 512
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.unk_token_id = tokenizer.unk_token_id   
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.unk_token_id = tokenizer.unk_token_id
    model.config.pad_token_id = tokenizer.pad_token_id


    # CSV dataset -> OnlineDPOTrainer expects a string column named "prompt"
    csv_path_abs = os.path.abspath(os.path.join(os.path.dirname(__file__), script_args.csv_path))
    ds = load_dataset("csv", data_files=csv_path_abs, split="train")

    if script_args.prompt_column != "prompt":
        ds = ds.rename_column(script_args.prompt_column, "prompt")

    ds = ds.map(lambda x: {"prompt": str(x["prompt"]).strip()})
    ds = ds.filter(lambda x: x["prompt"] is not None and x["prompt"] != "")
    # filter out prompts that are too long
    ds = ds.filter(lambda x: len(x["prompt"]) <= 512)
    # filter out prompts that are too short
    ds = ds.filter(lambda x: len(x["prompt"]) >= 8)
    ds = ds.remove_columns([c for c in ds.column_names if c != "prompt"])

    # Eval dataset (optional)
    
    # Eval dataset (optional)
    eval_dataset = ds.select(range(min(256, len(ds))))
    if len(eval_dataset) == 0:
        raise ValueError("eval_dataset is empty after filtering; cannot log completions.")

    # Debug: print eval dataset info
    if accelerator.is_main_process:
        print(f"DEBUG: eval_dataset length = {len(eval_dataset)}")
        print(f"DEBUG: eval_dataset columns = {eval_dataset.column_names}")
        if len(eval_dataset) > 0:
            print(f"DEBUG: first eval prompt sample = {eval_dataset[0]}")

    # LoRA (passed to trainer, not to from_pretrained)
    peft_config = None
    if script_args.use_lora:
        peft_config = LoraConfig(
            r=script_args.lora_rank,
            lora_alpha=2 * script_args.lora_rank,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

    reward_funcs = [AutoModelForSequenceClassification.from_pretrained(path, num_labels=1, torch_dtype=torch.bfloat16) for path in reward_model_paths]
    reward_processing_classes = [AutoTokenizer.from_pretrained(path) for path in reward_model_paths]
    
    trainer = OnlineDPOTrainer(
        model=model,
        ref_model=None,
        reward_funcs=reward_funcs,
        judge=None,
        args=training_args,
        train_dataset=ds,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        reward_processing_classes = reward_processing_classes,
        peft_config=peft_config,
    )

    if (not script_args.disable_wandb) and accelerator.is_main_process:
        wandb.init(project="moodpo", name=script_args.wandb_name)

    generation_config = GenerationConfig(
        max_new_tokens=training_args.max_new_tokens,
        do_sample=True,
        temperature=training_args.temperature
    )

    # Ensure report_to includes wandb for callback to work
    if not script_args.disable_wandb and "wandb" not in training_args.report_to:
        training_args.report_to = ["wandb"]

    if (not script_args.disable_wandb) and accelerator.is_main_process:
        wandb.config.update(vars(script_args))
        wandb.config.update(vars(training_args))
    
    trainer.train()
    trainer.save_model(training_args.output_dir)