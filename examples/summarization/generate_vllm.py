import gc
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from datasets import Dataset, DatasetInfo, builder, load_dataset
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

from trl import DPOTrainer


builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True


@dataclass
class ScriptArguments:
    output_dir: Optional[str] = field(
        default="compare_results",
        metadata={"help": "output folder"},
    )
    num_gpus: Optional[int] = field(default=1)
    model_name: Optional[str] = field(default="EleutherAI/pythia-410m", metadata={"help": "the model name"})
    revision: Optional[str] = field(default="main", metadata={"help": "the model revision"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "the tokenizer name"})
    dataset_name: Optional[str] = field(
        default="arianhosseini/openai_summarize_unlabelled", metadata={"help": "the dataset name"}
    )
    train_split: Optional[str] = field(default="train[:20]", metadata={"help": "the dataset name"})
    batch_size: Optional[int] = field(default=4)
    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})

    sample_n: Optional[int] = field(default=2, metadata={"help": "Gen temperature"})
    temperature: Optional[float] = field(default=0.7, metadata={"help": "Gen temperature"})
    top_p: Optional[float] = field(default=1.0, metadata={"help": "Gen temperature"})
    max_new_tokens: Optional[int] = field(default=48, metadata={"help": "max new tokens"})
    dtype: Optional[str] = field(default="auto")

    lora_model: Optional[bool] = field(default=False)
    base_model_name: Optional[str] = field(default=None, metadata={"help": "the model name"})
    base_model_revision: Optional[str] = field(default=None)
    ref_model_name: Optional[str] = field(default=None, metadata={"help": "the model name"})
    ref_model_revision: Optional[str] = field(default=None)


def prepare_vllm_model(script_args):
    if script_args.tokenizer_name is not None:
        tokenizer_name = script_args.tokenizer_name
    else:
        tokenizer_name = script_args.model_name

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if tokenizer_name.startswith("EleutherAI"):
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    tokenizer.padding_side = "left"

    if script_args.lora_model:
        # peft model that needs to be merged
        if script_args.base_model_name is not None:
            base_model = AutoModelForCausalLM.from_pretrained(
                script_args.base_model_name, revision=script_args.base_model_revision
            )
            model = PeftModelForCausalLM.from_pretrained(
                base_model, script_args.model_name, revision=script_args.revision, device="cpu"
            )
        else:
            model = AutoPeftModelForCausalLM.from_pretrained(
                script_args.model_name, revision=script_args.revision, device="cpu"
            )
        # merge the model and save
        merged = model.merge_and_unload()
        model_save_path = f"/home/toolkit/trl_results/{script_args.model_name}_merged/{script_args.revision}"
        merged.save_pretrained(model_save_path)
        del model
        del merged
        model_name = model_save_path
        revision = None
    else:
        model_name = script_args.model_name
        revision = script_args.revision

    llm = LLM(
        model=model_name,
        revision=revision,
        dtype=script_args.dtype,
        tokenizer=tokenizer_name,
        tensor_parallel_size=script_args.num_gpus,
        trust_remote_code=True,
    )
    llm.set_tokenizer(tokenizer)

    return llm, tokenizer


def strip_prompt(examples):
    examples["prompt"] = [prompt.strip() for prompt in examples["prompt"]]

    return examples


def generate_vllm(script_args):
    llm, _ = prepare_vllm_model(script_args)

    dataset = load_dataset(script_args.dataset_name, split=script_args.train_split)

    prompts = dataset["query"]

    sampling_params = SamplingParams(
        temperature=script_args.temperature,
        max_tokens=script_args.max_new_tokens,
        top_p=script_args.top_p,
        n=script_args.sample_n,
    )

    generations = llm.generate(prompts, sampling_params)

    print(f"generated {len(generations)} samples")

    def dataset_generator():
        for gen in generations:
            if len(gen.outputs) == 2:
                yield {
                    "prompt": gen.prompt,
                    "chosen": gen.outputs[0].text,
                    "rejected": gen.outputs[1].text,
                }
            else:
                row = {"prompt": gen.prompt}
                for i, output in enumerate(gen.outputs):
                    row[f"target{i}"] = output.text

                yield row

    ds_info = DatasetInfo(
        f"{script_args.dataset_name} split {script_args.train_split} prompts used to generate with {script_args.model_name}"
        f" temp {script_args.temperature} top_p {script_args.top_p} "
    )
    generated_dataset = Dataset.from_generator(dataset_generator, info=ds_info)

    destroy_model_parallel()
    del llm.llm_engine.driver_worker
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.destroy_process_group()

    return generated_dataset


def relabel(script_args, dataset):
    torch_dtype = script_args.dtype if script_args.dtype in ["auto", None] else getattr(torch, script_args.dtype)

    if script_args.base_model_name is not None:
        base_model = AutoModelForCausalLM.from_pretrained(
            script_args.base_model_name,
            revision=script_args.base_model_revision,
            torch_dtype=torch_dtype,
        )
        model = PeftModelForCausalLM.from_pretrained(
            base_model,
            script_args.model_name,
            revision=script_args.revision,
            torch_dtype=torch_dtype,
        )
        ref_model = None
    elif script_args.lora_model:
        model = AutoPeftModelForCausalLM.from_pretrained(
            script_args.model_name,
            revision=script_args.revision,
            torch_dtype=torch_dtype,
        )
        ref_model = None
    else:
        assert script_args.ref_model is not None
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_name,
            revision=script_args.revision,
            torch_dtype=torch_dtype,
        )
        ref_model = AutoModelForCausalLM.from_pretrained(
            script_args.ref_model_name,
            revision=script_args.ref_model_revision,
            torch_dtype=torch_dtype,
        )

    if script_args.tokenizer_name is not None:
        tokenizer_name = script_args.tokenizer_name
    else:
        tokenizer_name = script_args.model_name

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if tokenizer_name.startswith("EleutherAI"):
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    training_args = TrainingArguments(per_device_eval_batch_size=int(script_args.batch_size), output_dir=".")

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        args=training_args,
        max_length=script_args.max_new_tokens + script_args.max_prompt_length,
        max_target_length=script_args.max_new_tokens,
        max_prompt_length=script_args.max_prompt_length,
    )

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

    dpo_trainer.accelerator.print("Prediction")
    preds, _, metrics = dpo_trainer.predict(dataset)
    (
        chosen_rewards,
        rejected_rewards,
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
    ) = preds
    dpo_trainer.accelerator.print(f"metrics {metrics}")

    if dpo_trainer.accelerator.is_local_main_process:
        print("Relabelling Dataset and Saving")
        dataset = dataset.add_column("pred_chosen", chosen_rewards)
        dataset = dataset.add_column("pred_rejected", rejected_rewards)

        relabel_dataset = dataset.map(
            relabel_with_preds,
            batched=True,
        )

        description = f"{script_args.dataset_name} relabelled with {script_args.model_name}"
        relabel_dataset._info.description = description
        relabel_dataset.push_to_hub(os.path.basename(script_args.output_dir), split="train")


def generate_relabel_args_dict(args_dict):
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_dict(args_dict)[0]
    dataset = generate_vllm(script_args)
    relabel(script_args, dataset)


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    generate_vllm(script_args)
