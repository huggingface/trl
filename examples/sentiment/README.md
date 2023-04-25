# Sentiment Examples

The notebooks and scripts in this examples show how to fine-tune a model with a sentiment classifier (such as `lvwerra/distilbert-imdb`).

Here's an overview of the notebooks and scripts:

| File | Description |
|---|---|
| `notebooks/gpt2-sentiment.ipynb`  | Fine-tune GPT2 to generate positive movie reviews. |
| `notebooks/gpt2-sentiment-control.ipynb`  | Fine-tune GPT2 to generate movie reviews with controlled sentiment. |
| `scripts/gpt2-sentiment.py` | Same as the notebook, but easier to use to use in mutli-GPU setup. |
| `scripts/t5-sentiment.py` | Same as GPT2 script, but for a Seq2Seq model (T5). |

## Launch scripts

The `trl` library is powered by `accelerate`. As such it is best to configure and launch trainings with the following commands:

```bash
accelerate config # will prompt you to define the training configuration
accelerate launch scripts/gpt2-sentiment.py # launches training
```


