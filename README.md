<div style="text-align: center">
<img src="https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/images/trl_banner_dark.png">
</div>

# TRL - Transformer Reinforcement Learning
> Train transformer language models with reinforcement learning.


## What is it?
With `trl` you can train transformer language models with Proximal Policy Optimization (PPO). The library is built on top of the [`transformers`](https://github.com/huggingface/transformers) library by  🤗 Hugging Face. Therefore, pre-trained language models can be directly loaded via `transformers`. At this point most of decoder architectures and encoder-decoder architectures are supported. 

**Highlights:**
- `PPOTrainer`: A PPO trainer for language models that just needs (query, response, reward) triplets to optimise the language model.
- `AutoModelForCausalLMWithValueHead` & `AutoModelForSeq2SeqLMWithValueHead`: A transformer model with an additional scalar output for each token which can be used as a value function in reinforcement learning.
- Example: Train GPT2 to generate positive movie reviews with a BERT sentiment classifier.

## How it works
Fine-tuning a language model via PPO consists of roughly three steps:

1. **Rollout**: The language model generates a response or continuation based on query which could be the start of a sentence.
2. **Evaluation**: The query and response are evaluated with a function, model, human feedback or some combination of them. The important thing is that this process should yield a scalar value for each query/response pair.
3. **Optimization**: This is the most complex part. In the optimisation step the query/response pairs are used to calculate the log-probabilities of the tokens in the sequences. This is done with the model that is trained and and a reference model, which is usually the pre-trained model before fine-tuning. The KL-divergence between the two outputs is used as an additional reward signal to make sure the generated responses don't deviate to far from the reference language model. The active language model is then trained with PPO.

This process is illustrated in the sketch below:


<div style="text-align: center">
<img src="https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/images/trl_overview.png" width="800">
<p style="text-align: center;"> <b>Figure:</b> Sketch of the workflow. </p>
</div>

## Installation

### Python package
Install the library with pip:
```bash
pip install trl
```

### From source
If you want to run the examples in the repository a few additional libraries are required. Clone the repository and install it with pip:
```bash
git clone https://github.com/lvwerra/trl.git
cd trl/
pip install .
```

If you wish to develop TRL, you should install in editable mode:
```bash
pip install -e .
```

## How to use

### Example
This is a basic example on how to use the library. Based on a query the language model creates a response which is then evaluated. The evaluation could be a human in the loop or another model's output.

```python
# imports
import torch
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from trl.core import respond_to_batch

# get models
model = AutoModelForCausalLMWithValueHead.from_pretrained('gpt2')
model_ref = create_reference_model(model)

tokenizer = AutoTokenizer.from_pretrained('gpt2')

# initialize trainer
ppo_config = PPOConfig(
    batch_size=1,
)

# encode a query
query_txt = "This morning I went to the "
query_tensor = tokenizer.encode(query_txt, return_tensors="pt")

# get model response
response_tensor  = respond_to_batch(model, query_tensor)

# create a ppo trainer
ppo_trainer = PPOTrainer(ppo_config, model, model_ref, tokenizer)

# define a reward for response
# (this could be any reward such as human feedback or output from another model)
reward = [torch.tensor(1.0)]

# train model for one step with ppo
train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)
```

### Advanced example: IMDB sentiment
For a detailed example check out the example python script `examples/sentiment/scripts/gpt2-sentiment.py`, where GPT2 is fine-tuned to generate positive movie reviews. An few examples from the language models before and after optimisation are given below:

<div style="text-align: center">
<img src="https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/images/table_imdb_preview.png" width="800">
<p style="text-align: center;"> <b>Figure:</b> A few review continuations before and after optimisation. </p>
</div>

## References

### Proximal Policy Optimisation
The PPO implementation largely follows the structure introduced in the paper **"Fine-Tuning Language Models from Human Preferences"** by D. Ziegler et al. \[[paper](https://arxiv.org/pdf/1909.08593.pdf), [code](https://github.com/openai/lm-human-preferences)].

### Language models
The language models utilize the `transformers` library by 🤗 Hugging Face.

## Citation

```bibtex
@misc{vonwerra2022trl,
  author = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert},
  title = {TRL: Transformer Reinforcement Learning},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/lvwerra/trl}}
}
```
