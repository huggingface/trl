# DPO pipeline for the creation of StackLlaMa 2: a Stack exchange llama-v2-7b model

## Prerequisites

Install all the dependencies in the `requirements.txt`:

```
$ pip install -U -r requirements.txt
```

Since we will use `accelerate` for training, make sure to run:
```
$ accelerate config
```

## Training

There were two main steps to the DPO training process:
1. Supervised fine-tuning of the base llama-v2-7b model to create llama-v2-7b-se:
    - `accelerate launch examples/stack_llama_2/scripts/sft_llama2.py --output_dir="sft"`
1. Run the DPO trainer using the model saved by the previous step:
    - `accelerate launch examples/stack_llama_2/scripts/dpo_llama2.py --model_name_or_path="sft/final_checkpoint" --output_dir="dpo"`


## Merging the adaptors

To merge the adaptors into the base model we can use the `merge_peft_adapter.py` helper script that comes with TRL:

```
python trl/examples/research_projects/stack_llama/scripts/merge_peft_adapter.py --base_model_name="meta-llama/Llama-2-7b-hf" --adapter_model_name="dpo/final_checkpoint/" --output_name="stack-llama-2"
```

which will also push the model to your HuggingFace hub account.

## Running the model

We can load the DPO-trained LoRA adaptors which were saved by the DPO training step and load them via:

```py
from peft import AutoPeftModelForCausalLM


model = AutoPeftModelForCausalLM.from_pretrained(
    "dpo/final_checkpoint",
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
)

model.generate(...)
```

## for google colab

Make sure we use T5 GPU.
Unfortunately, the T5 is not an Ampere architecture. So we need to change bf16 to fp16.

```
!git clone https://github.com/huggingface/trl.git
!pip install -q -U -r trl/examples/research_projects/stack_llama_2/scripts/requirements.txt
!find trl/examples/research_projects/stack_llama_2/scripts -name "sft_llama2.py" -exec sed -i "s/bf16/fp16/g" {} \;
!accelerate config # This machine, No distributed training, NO, NO, NO, all, fp16
!huggingface-cli login # put access Token
!accelerate launch trl/examples/research_projects/stack_llama_2/scripts/sft_llama2.py --output_dir="sft"
```
