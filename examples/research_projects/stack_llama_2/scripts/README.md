# DPO pipeline for the creation of StackLlaMa 2: a Stack exchange llama-v2-7b model

## Prerequisites

Install all the dependencies in the `requirements.txt`:

```
$ pip install -U -r requirements.txt
```

## Training

There were two main steps to the training process:
1. Supervised fine-tuning of the base llama-v2-7b model to create llama-v2-7b-se:
    - `torchrun --nnodes 1  --nproc_per_node 8 examples/stack_llama_2/scripts/sft_llama2.py`
1. Run the DPO trainer:
    - `torchrun --nnodes 1  --nproc_per_node 8 examples/stack_llama_2/scripts/sft_llama2.py`
