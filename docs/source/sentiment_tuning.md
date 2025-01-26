# Sentiment Tuning Examples

The notebooks and scripts in this examples show how to fine-tune a model with a sentiment classifier (such as `lvwerra/distilbert-imdb`).

Here's an overview of the notebooks and scripts in the [trl repository](https://github.com/huggingface/trl/tree/main/examples):



| File                                                                                           | Description                                                                                                              |
|------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| [`examples/scripts/ppo.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/ppo.py)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/sentiment/notebooks/gpt2-sentiment.ipynb) | This script shows how to use the `PPOTrainer` to fine-tune a sentiment analysis model using IMDB dataset                 |
| [`examples/notebooks/gpt2-sentiment.ipynb`](https://github.com/huggingface/trl/tree/main/examples/notebooks/gpt2-sentiment.ipynb)              | This notebook demonstrates how to reproduce the GPT2 imdb sentiment tuning example on a jupyter notebook.                |
| [`examples/notebooks/gpt2-control.ipynb`](https://github.com/huggingface/trl/tree/main/examples/notebooks/gpt2-control.ipynb)   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/sentiment/notebooks/gpt2-sentiment-control.ipynb)                | This notebook demonstrates how to reproduce the GPT2 sentiment control example on a jupyter notebook.    



## Usage

```bash
# 1. run directly
python examples/scripts/ppo.py
# 2. run via `accelerate` (recommended), enabling more features (e.g., multiple GPUs, deepspeed)
accelerate config # will prompt you to define the training configuration
accelerate launch examples/scripts/ppo.py # launches training
# 3. get help text and documentation
python examples/scripts/ppo.py --help
# 4. configure logging with wandb and, say, mini_batch_size=1 and gradient_accumulation_steps=16
python examples/scripts/ppo.py --log_with wandb --mini_batch_size 1 --gradient_accumulation_steps 16
```

Note: if you don't want to log with `wandb` remove `log_with="wandb"` in the scripts/notebooks. You can also replace it with your favourite experiment tracker that's [supported by `accelerate`](https://huggingface.co/docs/accelerate/usage_guides/tracking).


## Few notes on multi-GPU 

To run in multi-GPU setup with DDP (distributed Data Parallel) change the `device_map` value to `device_map={"": Accelerator().process_index}` and make sure to run your script with `accelerate launch yourscript.py`. If you want to apply naive pipeline parallelism you can use `device_map="auto"`.