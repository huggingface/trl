import gc
import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from datasets import Dataset, builder, load_dataset
from huggingface_hub import list_repo_refs
from peft import PeftModelForCausalLM
from scalar_rm_model import ScalarModel, ScalarModelConfig
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

import wandb


builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True


@dataclass
class GenerateScriptArguments:
    output_dir: Optional[str] = field(
        default="/home/toolkit/trl_results",
        metadata={"help": "output folder"},
    )
    num_gpus: Optional[int] = field(default=1)
    base_model_name: Optional[str] = field(default=None, metadata={"help": "the model name"})
    base_model_revision: Optional[str] = field(default=None)
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
    eval_batch_size: Optional[int] = field(default=16)
    max_length: Optional[int] = field(default=512)
    gold_tokenizer_name: Optional[str] = field(default=None, metadata={"help": "the tokenizer name"})
    flash_attention: Optional[bool] = field(default=False)


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
        skip_special_tokens=False,
    )

    refs = list_repo_refs(script_args.model_name, repo_type="model")
    gens = {}
    revisions = sorted([branch.name for branch in refs.branches])
    for revision in revisions:
        if revision == "main":
            continue

        if script_args.model_revisions and revision not in script_args.model_revisions:
            continue

        print(f"generating step {revision}")

        if script_args.base_model_name is None:
            # merged model
            model_name = script_args.model_name
            revision_name = revision
        else:
            # peft model that needs to be merged
            base_model = AutoModelForCausalLM.from_pretrained(
                script_args.base_model_name, revision=script_args.base_model_revision
            )
            # merge the model and save
            model = PeftModelForCausalLM.from_pretrained(
                base_model, script_args.model_name, revision=revision, device="cpu"
            )
            merged = model.merge_and_unload()
            model_save_path = f"/home/toolkit/trl_results/{script_args.model_name}_merged/{revision}"
            merged.save_pretrained(model_save_path)
            del model
            del merged
            model_name = model_save_path
            revision_name = revision
            revision = None

        llm = LLM(
            model=model_name,
            revision=revision,
            tokenizer=script_args.tokenizer_name,
            dtype=script_args.gen_dtype,
            max_model_len=script_args.seq_length,
            tensor_parallel_size=script_args.num_gpus,
            trust_remote_code=True,
        )

        llm.set_tokenizer(tokenizer)

        generations = llm.generate(prompts, sampling_params)

        texts = [output.prompt + output.outputs[0].text for output in generations]

        gens[revision_name] = texts

        dataset = dataset.add_column(f"generations_{revision_name}", texts)

        # delete old model
        destroy_model_parallel()
        del llm.llm_engine.driver_worker
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        torch.distributed.destroy_process_group()

    if script_args.output_dir is not None:
        # TODO add hash to dataset path
        # sampling_str = str(sampling_params)
        # sampling_hash = hashlib.sha256(sampling_str.encode()).hexdigest()[:10]
        dataset_path = os.path.join(
            script_args.output_dir,
            script_args.dataset_name.replace("/", "_"),
            script_args.model_name.replace("/", "_"),
        )
        os.makedirs(dataset_path, exist_ok=True)
        dataset.save_to_disk(dataset_path)
        with open(f"{dataset_path}_sampling_params.txt", "w") as f:
            print(sampling_params, file=f)

    print(f"generated {len(gens)} steps")
    reference = dataset["query_reference_response"]

    return reference, gens

    # ds_info = DatasetInfo(
    #     f"{script_args.dataset_name} split {script_args.train_split} prompts used to generate with {script_args.model_name}"
    #     f" temp {script_args.temperature} top_p {script_args.top_p} "
    # )
    # generated_dataset = Dataset.from_generator(dataset_generator, info=ds_info)
    # generated_dataset.push_to_hub(os.path.basename(script_args.output_dir), split="train")


def evaluate(args, reference, generations, model_name=None):
    if args.wandb_log_id is not None:
        # don't overwrite the wandb name of the original run
        if args.wandb_log_id == "model_name":
            # model name = config_wandblogid
            wandb_log_id = model_name.split("_")[-1]
        else:
            wandb_log_id = args.wandb_log_id

        os.environ.pop("WANDB_NAME")
        # original_name = wandb_name.removeprefix("geneval_")
        wandb.init(id=wandb_log_id, resume="allow")
        log_to_wandb = True
        print(f"Logging to WandB {wandb_log_id}")
    else:
        log_to_wandb = False

    torch_dtype = args.eval_dtype if args.eval_dtype in ["auto", None] else getattr(torch, args.eval_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.gold_tokenizer_name)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    if args.gold_model_name.startswith("vwxyzjn"):
        # ScalarModel
        scalar_model_config = ScalarModelConfig.from_pretrained(
            args.gold_model_name,
            revision=args.gold_model_revision,
        )
        # hack to remove the path
        # models/EleutherAI/pythia-6.9b-deduped/sft_model_55513 -> EleutherAI/pythia-6.9b-deduped
        if scalar_model_config.base_model.startswith("models/"):
            original_model = scalar_model_config.base_config["_name_or_path"].split("/")[2]
            sft_model = f"vwxyzjn/EleutherAI_{original_model}__sft__tldr"
            scalar_model_config.base_config["_name_or_path"] = sft_model
            scalar_model_config.base_model = sft_model
            _, seed, _ = args.gold_model_revision.split("__")
            scalar_model_config.base_model_revision = f"sft__{seed}__1708611267"

        # quantization_config = get_quantization_config(model_config)
        model = ScalarModel.from_pretrained(
            args.gold_model_name,
            revision=args.gold_model_revision,
            config=scalar_model_config,
            torch_dtype=torch_dtype,
            use_flash_attention_2=args.flash_attention,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.gold_model_name,
            revision=args.gold_model_revision,
            torch_dtype=torch_dtype,
        )

    model.config.pad_token_id = tokenizer.pad_token_id

    training_args = TrainingArguments(per_device_eval_batch_size=int(args.eval_batch_size), output_dir=".")

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
    )

    def tokenize_and_add_eos(tokenizer, text_column, max_length):
        def fn(example):
            text = example[text_column]
            ends_with_eos = text.endswith(tokenizer.eos_token)
            if not ends_with_eos:
                text += tokenizer.eos_token

            tokenized = tokenizer(
                text,
                padding="max_length",
                max_length=max_length,
                truncation=True,
            )

            # guarantee that last token is EOS if truncated
            token_length = sum(tokenized["attention_mask"])
            if token_length == max_length:
                tokenized["input_ids"][-1] = tokenizer.eos_token_id

            return tokenized

        return fn

    ## get reference continuation rewards
    dataset = Dataset.from_dict({"reference": reference})
    dataset = dataset.map(tokenize_and_add_eos(tokenizer, "reference", args.max_length))

    ref_results = trainer.predict(dataset)
    ref_rewards = ref_results.predictions[0]

    step = 0
    for step_str, query_response in generations.items():
        dataset = Dataset.from_dict({"query_response": query_response})
        dataset = dataset.map(tokenize_and_add_eos(tokenizer, "query_response", args.max_length))

        print(f"Evaluating {step_str}")
        results = trainer.predict(dataset)
        gen_rewards = results.predictions[0]

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
                    "train/global_step": step,
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
    # dataset = load_dataset(generate_args.dataset_name, split=generate_args.split)
    # generations = {"step0": dataset["query_reference_response"]}
    # reference = dataset["query_reference_response"]
    print("EVALUATING")
    evaluate(eval_args, reference, generations, generate_args.model_name)


if __name__ == "__main__":
    parser = HfArgumentParser([GenerateScriptArguments, EvalScriptArguments])
    generate_args, eval_args = parser.parse_args_into_dataclasses()
    if eval_args.gold_tokenizer_name is None:
        eval_args.gold_tokenizer_name = generate_args.tokenizer_name

    print("GENERATING")
    reference, generations = generate(generate_args)
    # dataset = load_dataset(generate_args.dataset_name, split=generate_args.train_split)
    # generations = {"step0": dataset["query_reference_response"]}
    # reference = dataset["query_reference_response"]
    print("EVALUATING")
    evaluate(eval_args, reference, generations)
