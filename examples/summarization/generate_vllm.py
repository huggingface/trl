import os
from dataclasses import dataclass, field
from typing import Optional

from datasets import Dataset, DatasetInfo, load_dataset
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
)
from vllm import LLM, SamplingParams


@dataclass
class ScriptArguments:
    output_dir: Optional[str] = field(
        default="compare_results",
        metadata={"help": "output folder"},
    )
    num_gpus: Optional[int] = field(default=1)
    model_name: Optional[str] = field(default="EleutherAI/pythia-410m", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "the tokenizer name"})
    dataset_name: Optional[str] = field(
        default="arianhosseini/openai_summarize_unlabelled", metadata={"help": "the dataset name"}
    )
    train_split: Optional[str] = field(default="train[:20]", metadata={"help": "the dataset name"})
    batch_size: Optional[int] = field(default=4)
    seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})

    temperature: Optional[float] = field(default=0.7, metadata={"help": "Gen temperature"})
    top_p: Optional[float] = field(default=1.0, metadata={"help": "Gen temperature"})
    max_new_tokens: Optional[int] = field(default=48, metadata={"help": "max new tokens"})


def prepare_vllm_model(script_args):
    if script_args.tokenizer_name is not None:
        tokenizer_name = script_args.tokenizer_name
    else:
        tokenizer_name = script_args.model_name

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    tokenizer.padding_side = "left"

    llm = LLM(
        model=script_args.model_name,
        dtype="auto",
        tokenizer=tokenizer_name,
        max_model_len=script_args.seq_length,
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
    dataset = dataset.map(strip_prompt, batched=True)

    prompts = dataset["prompt"]

    sampling_params = SamplingParams(
        temperature=script_args.temperature,
        max_tokens=script_args.max_new_tokens,
        top_p=script_args.top_p,
        n=2,
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
                print("skipping gen, only 1 output")

    ds_info = DatasetInfo(
        f"{script_args.dataset_name} split {script_args.train_split} prompts used to generate with {script_args.model_name}"
        f" temp {script_args.temperature} top_p {script_args.top_p} "
    )
    generated_dataset = Dataset.from_generator(dataset_generator, info=ds_info)
    generated_dataset.push_to_hub(os.path.basename(script_args.output_dir), split="train")


def generate_vllm_args_dict(args_dict):
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_dict(args_dict)[0]
    generate_vllm(script_args)


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    generate_vllm(script_args)
