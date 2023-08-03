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
    - `accelerate launch examples/stack_llama_2/scripts/sft_llama2.py`
1. Run the DPO trainer using the model saved by the previous step:
    - `accelerate launch examples/stack_llama_2/scripts/dpo_llama2.py`
