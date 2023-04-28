# Summarization Examples
  
The script in this example show how to train a reward model for summarization, following the OpenAI Learning to Summarize from Human Feedback [paper](https://arxiv.org/abs/2009.01325). We've validated that the script can be used to train a small GPT2 to get slightly over 60% validation accuracy, which is aligned with results from the paper. The model is [here](https://huggingface.co/Tristan/gpt2_reward_summarization).

Here's an overview of the files:

| File | Description |
|---|---|
| `scripts/reward_summarization.py` | For tuning the reward model. |
| `scripts/ds3_reward_summarization_example_config.json` | Can be used with the reward model script to scale it up to arbitrarily big models that don't fit on a single GPU. |


## Installation

```bash
pip install trl
pip install evaluate
# optional: deepspeed
pip install deepspeed
```

```bash
# If you want your reward model to follow the Learning to Summarize from Human Feedback paper closely, then tune a GPT model on summarization and then instantiate the reward model
# with it. In other words, pass in the name of your summarization-finetuned gpt on the hub, instead of the name of the pretrained gpt2 like we do in the following examples of how
# to run this script.

# Example of running this script with the small size gpt2 on a 40GB A100 (A100's support bf16). Here, the global batch size will be 64:
python -m torch.distributed.launch --nproc_per_node=1 reward_summarization.py --bf16

# Example of running this script with the xl size gpt2 on 16 40GB A100's. Here the global batch size will still be 64:
python -m torch.distributed.launch --nproc_per_node=16 reward_summarization.py --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=4 --gpt_model_name=gpt2-xl --bf16 --deepspeed=ds3_reward_summarization_example_config.json
```
