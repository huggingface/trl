import gc
import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from accelerate import Accelerator
from datasets import Dataset, DatasetInfo, builder, load_dataset
from huggingface_hub import list_repo_refs
from scalar_rm_model import ScalarModel, ScalarModelConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, HfArgumentParser, Trainer
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

import wandb


builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True


@dataclass
class GenerateScriptArguments:
    output_dir: Optional[str] = field(
        default="compare_results",
        metadata={"help": "output folder"},
    )
    num_gpus: Optional[int] = field(default=1)
    model_name: Optional[str] = field(default="EleutherAI/pythia-410m", metadata={"help": "the model name"})
    model_revisions: Optional[List[str]] = field(default_factory=list)
    # base_model_revision: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "the tokenizer name"})
    dataset_name: Optional[str] = field(
        default="arianhosseini/openai_summarize_unlabelled", metadata={"help": "the dataset name"}
    )
    split: Optional[str] = field(default="validation", metadata={"help": "the dataset name"})
    batch_size: Optional[int] = field(default=4)
    seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})

    temperature: Optional[float] = field(default=0.7, metadata={"help": "Gen temperature"})
    top_p: Optional[float] = field(default=1.0, metadata={"help": "Gen temperature"})
    max_new_tokens: Optional[int] = field(default=48, metadata={"help": "max new tokens"})
    gen_dtype: Optional[str] = field(default="auto")


@dataclass
class EvalScriptArguments:
    wandb_log_id: Optional[str] = field(default=None)
    gold_model_name: Optional[str] = field(default="EleutherAI/pythia-410m", metadata={"help": "the model name"})
    gold_model_revision: Optional[str] = field(default=None)
    eval_dtype: Optional[str] = field(default="auto")
    max_length: Optional[int] = field(default=512)
    gold_tokenizer_name: Optional[str] = field(default=None, metadata={"help": "the tokenizer name"})


def generate(script_args):
    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "left"

    dataset = load_dataset(script_args.dataset_name, split=script_args.split)
    prompts = dataset["query"]

    sampling_params = SamplingParams(
        temperature=script_args.temperature,
        max_tokens=script_args.max_new_tokens,
        top_p=script_args.top_p,
        n=1,
        include_stop_str_in_output=True,
    )

    refs = list_repo_refs(script_args.model_name, repo_type="model")
    gens = {}
    for branch in refs.branches:
        if branch.name == "main":
            continue

        if script_args.model_revisions and branch.name not in script_args.model_revisions:
            continue

        print(f"generating step {branch.name}")
        llm = LLM(
            model=script_args.model_name,
            tokenizer=script_args.tokenizer_name,
            revision=branch.name,
            dtype=script_args.gen_dtype,
            max_model_len=script_args.seq_length,
            tensor_parallel_size=script_args.num_gpus,
            trust_remote_code=True,
        )

        llm.set_tokenizer(tokenizer)

        generations = llm.generate(prompts, sampling_params)

        texts = [output.prompt + output.outputs[0].text for output in generations]

        gens[branch.name] = texts

        # delete old model
        destroy_model_parallel()
        del llm.llm_engine.driver_worker
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        torch.distributed.destroy_process_group()

    reference = dataset["query_reference_response"]

    print(f"generated {len(gens)} steps")
    return reference, gens

    # ds_info = DatasetInfo(
    #     f"{script_args.dataset_name} split {script_args.train_split} prompts used to generate with {script_args.model_name}"
    #     f" temp {script_args.temperature} top_p {script_args.top_p} "
    # )
    # generated_dataset = Dataset.from_generator(dataset_generator, info=ds_info)
    # generated_dataset.push_to_hub(os.path.basename(script_args.output_dir), split="train")


def evaluate(script_args, reference, generations):
    if script_args.wandb_log_id is not None:
        wandb_name = os.environ["WANDB_NAME"]
        original_name = wandb_name.removeprefix("geneval_")
        wandb.init(id=script_args.wandb_log_id, resume="allow", name=original_name)
        log_to_wandb = True
        print("Logging to WandB")
    else:
        log_to_wandb = False

    torch_dtype = (
        script_args.eval_dtype if script_args.eval_dtype in ["auto", None] else getattr(torch, script_args.eval_dtype)
    )
    tokenizer = AutoTokenizer.from_pretrained(script_args.gold_tokenizer_name)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    scalar_model_config = ScalarModelConfig.from_pretrained(
        script_args.gold_model_name,
        revision=script_args.gold_model_revision,
        trust_remote_code=True,
    )
    # hack to remove the path
    # models/EleutherAI/pythia-6.9b-deduped/sft_model_55513 -> EleutherAI/pythia-6.9b-deduped
    if scalar_model_config.base_model.startswith("models/"):
        original_model = scalar_model_config.base_config["_name_or_path"].split("/")[2]
        sft_model = f"vwxyzjn/EleutherAI_{original_model}__sft__tldr"
        scalar_model_config.base_config["_name_or_path"] = sft_model
        scalar_model_config.base_model = sft_model
        _, seed, _ = script_args.gold_model_revision.split("__")
        scalar_model_config.base_model_revision = f"sft__{seed}__1708611267"

    # quantization_config = get_quantization_config(model_config)
    model = ScalarModel.from_pretrained(
        script_args.gold_model_name,
        revision=script_args.gold_model_revision,
        config=scalar_model_config,
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
    )

    ## get reference continuation rewards
    dataset = Dataset.from_dict({"reference": reference})
    dataset = dataset.map(
        lambda example: tokenizer(
            example["reference"],
            padding="max_length",
            max_length=script_args.max_length,
            truncation=True,
        ),
        batched=True,
    )

    ref_results = trainer.predict(dataset)
    ref_rewards = ref_results.predictions

    step = 0
    for step_str, query_response in generations.items():
        dataset = Dataset.from_dict({"query_response": query_response})
        dataset = dataset.map(
            lambda example: tokenizer(
                example["query_response"],
                padding="max_length",
                max_length=script_args.max_length,
                truncation=True,
            ),
            batched=True,
        )

        print(f"Evaluating {step_str}")
        results = trainer.predict(dataset)
        gen_rewards = results.predictions

        win_rate = (gen_rewards > ref_rewards).mean().item()
        norm_reward = (gen_rewards - ref_rewards).mean().item()

        if step_str.startswith("step"):
            step_str = step_str.removeprefix("step")

        if step_str.isdigit():
            step = int(step_str)
        else:
            print(f"Warning step name {step_str} is not an integer")
            step = step + 1

        if log_to_wandb:
            wandb.log(
                {
                    "gold/win_rate": win_rate,
                    "gold/norm_reward": norm_reward,
                    "step": step,
                }
            )

        print(f"step {step}: win-rate {win_rate} norm-reward {norm_reward}")


def main_args_dict(args_dict):
    parser = HfArgumentParser([GenerateScriptArguments, EvalScriptArguments])
    generate_args, eval_args = parser.parse_dict(args_dict)
    if eval_args.gold_tokenizer_name is None:
        eval_args.gold_tokenizer_name = generate_args.tokenizer_name

    print("GENERATING")
    reference, generations = generate(generate_args)
    # dataset = load_dataset(generate_args.dataset_name, split=generate_args.train_split)
    # generations = {"step0": dataset["query_reference_response"]}
    # reference = dataset["query_reference_response"]
    print("EVALUATING")
    evaluate(eval_args, reference, generations)


if __name__ == "__main__":
    parser = HfArgumentParser(GenerateScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    main(script_args)
