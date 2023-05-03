# Examples

_The best place to learn about examples in TRL is our [docs page](https://huggingface.co/docs/trl/index)!_

## Installation

```bash
pip install trl
#optional: wandb
pip install wandb
```
Note: if you don't want to log with `wandb` remove `log_with="wandb"` in the scripts/notebooks. 
You can also replace it with your favourite experiment tracker that's [supported by `accelerate`](https://huggingface.co/docs/accelerate/usage_guides/tracking).

## Accelerate Config
For all the examples, you'll need to generate an `Accelerate` config with:

```shell
accelerate config # will prompt you to define the training configuration
```

Then, it is encouraged to launch jobs with `accelerate launch`!

## Categories
The examples are currently split over the following categories:

**1: [Sentiment](https://github.com/lvwerra/trl/tree/main/examples/sentiment)**: Fine-tune a model with a sentiment classification model.
**2: [StackOverflow](https://github.com/lvwerra/trl/tree/main/examples/stack_llama)**: Perform the full RLHF process (fine-tuning, reward model training, and RLHF) on StackOverflow data.
**3: [summarization](https://github.com/lvwerra/trl/tree/main/examples/summarization)**: Recreate OpenAI's [Learning to Summarize paper](https://proceedings.neurips.cc/paper/2020/file/1f89885d556929e98d3ef9b86448f951-Paper.pdf).
**4: [toxicity](https://github.com/lvwerra/trl/tree/main/examples/toxicity)**: Fine-tune a model to reduce the toxicity of its generations.
write about best-of-n as an alternative rlhf
**5: [best-of-n sampling](https://github.com/lvwerra/trl/tree/main/examples/best_of_n_sampling)**: Comparative demonstration of best-of-n sampling as a simpler (but relatively expensive) alternative to RLHF
