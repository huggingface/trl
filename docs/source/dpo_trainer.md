# DPO Trainer

[![All_models-DPO-blue](https://img.shields.io/badge/All_models-DPO-blue)](https://huggingface.co/models?other=dpo,trl) [![smol_course-Chapter_2-yellow](https://img.shields.io/badge/smol_course-Chapter_2-yellow)](https://github.com/huggingface/smol-course/tree/main/2_preference_alignment)

## Overview

TRL supports the Direct Preference Optimization (DPO) Trainer for training language models, as described in the paper [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://huggingface.co/papers/2305.18290) by [Rafael Rafailov](https://huggingface.co/rmrafailov), Archit Sharma, Eric Mitchell, [Stefano Ermon](https://huggingface.co/ermonste), [Christopher D. Manning](https://huggingface.co/manning), [Chelsea Finn](https://huggingface.co/cbfinn).

The abstract from the paper is the following:

> While large-scale unsupervised language models (LMs) learn broad world knowledge and some reasoning skills, achieving precise control of their behavior is difficult due to the completely unsupervised nature of their training. Existing methods for gaining such steerability collect human labels of the relative quality of model generations and fine-tune the unsupervised LM to align with these preferences, often with reinforcement learning from human feedback (RLHF). However, RLHF is a complex and often unstable procedure, first fitting a reward model that reflects the human preferences, and then fine-tuning the large unsupervised LM using reinforcement learning to maximize this estimated reward without drifting too far from the original model. In this paper we introduce a new parameterization of the reward model in RLHF that enables extraction of the corresponding optimal policy in closed form, allowing us to solve the standard RLHF problem with only a simple classification loss. The resulting algorithm, which we call Direct Preference Optimization (DPO), is stable, performant, and computationally lightweight, eliminating the need for sampling from the LM during fine-tuning or performing significant hyperparameter tuning. Our experiments show that DPO can fine-tune LMs to align with human preferences as well as or better than existing methods. Notably, fine-tuning with DPO exceeds PPO-based RLHF in ability to control sentiment of generations, and matches or improves response quality in summarization and single-turn dialogue while being substantially simpler to implement and train.

This post-training method was contributed by [Kashif Rasul](https://huggingface.co/kashif) and later refactored by [Quentin Gallouédec](https://huggingface.co/qgallouedec).

## Quick start

This example demonstrates how to train a language model using the [`DPOTrainer`] from TRL. We train a [Qwen 3 0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) model on the [UltraFeedback dataset](https://huggingface.co/datasets/openbmb/UltraFeedback).

```python
from trl import DPOTrainer
from datasets import load_dataset

trainer = DPOTrainer(
    model="Qwen/Qwen3-0.6B",
    train_dataset=load_dataset("trl-lib/ultrafeedback_binarized", split="train"),
)
trainer.train()
```

<iframe src="https://trl-lib-trackio.hf.space/?project=trl-documentation&metrics=train*&runs=dpo_qwen3-0.6B_capybara" style="width: 100%; min-width: 300px; max-width: 800px;" height="830" frameBorder="0"></iframe>

## Expected dataset type and format

DPO requires a [preference](dataset_formats#preference) dataset. The [`DPOTrainer`] is compatible with both [standard](dataset_formats#standard) and [conversational](dataset_formats#conversational) dataset formats. When provided with a conversational dataset, the trainer will automatically apply the chat template to the dataset.

```python
# Standard format
## Explicit prompt (recommended)
preference_example = {"prompt": "The sky is", "chosen": " blue.", "rejected": " green."}
# Implicit prompt
preference_example = {"chosen": "The sky is blue.", "rejected": "The sky is green."}

# Conversational format
## Explicit prompt (recommended)
preference_example = {"prompt": [{"role": "user", "content": "What color is the sky?"}],
                      "chosen": [{"role": "assistant", "content": "It is blue."}],
                      "rejected": [{"role": "assistant", "content": "It is green."}]}
## Implicit prompt
preference_example = {"chosen": [{"role": "user", "content": "What color is the sky?"},
                                 {"role": "assistant", "content": "It is blue."}],
                      "rejected": [{"role": "user", "content": "What color is the sky?"},
                                   {"role": "assistant", "content": "It is green."}]}
```

If your dataset is not in one of these formats, you can preprocess it to convert it into the expected format. Here is an example with the [Vezora/Code-Preference-Pairs](https://huggingface.co/datasets/Vezora/Code-Preference-Pairs) dataset:

```python
from datasets import load_dataset

dataset = load_dataset("Vezora/Code-Preference-Pairs")


def preprocess_function(example):
    return {
        "prompt": [{"role": "user", "content": example["input"]}],
        "chosen": [{"role": "assistant", "content": example["accepted"]}],
        "rejected": [{"role": "assistant", "content": example["rejected"]}],
    }


dataset = dataset.map(preprocess_function, remove_columns=["instruction", "input", "accepted", "ID"])

print(next(iter(dataset["train"])))
```

```json
{
    "prompt": [{"role": "user", "content": "Create a nested loop to print every combination of numbers [...]"}],
    "chosen": [{"role": "assistant", "content": "Here is an example of a nested loop in Python [...]"}],
    "rejected": [{"role": "assistant", "content": "Here is an example of a nested loop in Python [...]"}],
}
```

## Looking deeper into the DPO method

Direct Preference Optimization (DPO) is a training method designed to align a language model with preference data. Instead of supervised input–output pairs, the model is trained on pairs of completions to the same prompt, where one completion is preferred over the other. The objective directly optimizes the model to assign higher likelihood to preferred completions than to dispreferred ones, relative to a reference model, without requiring an explicit reward model.

This section breaks down how DPO works in practice, covering the key steps: **preprocessing** and **loss computation**.

### Preprocessing and tokenization

During training, each example is expected to contain a prompt along with a preferred (`chosen`) and a dispreferred (`rejected`) completion. For more details on the expected formats, see [Dataset formats](dataset_formats).
The [`DPOTrainer`] tokenizes each input using the model's tokenizer.

### Computing the loss

![dpo_figure](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/dpo_figure.png)

The loss used in DPO is defined as follows:
$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}_{(x,y^{+},y^{-})}\!\left[\log \sigma\!\left(\beta\Big(\log\frac{{\pi_{\theta}(y^{+}\!\mid x)}}{{\pi_{\mathrm{ref}}(y^{+}\!\mid x)}}-\log \frac{{\pi_{\theta}(y^{-}\!\mid x)}}{{\pi_{\mathrm{ref}}(y^{-}\!\mid x)}}\Big)\right)\right]
$$
  
where  \\( x \\)  is the prompt,  \\( y^+ \\) is the preferred completion and  \\( y^- \\)  is the dispreferred completion.  \\( \pi_{\theta} \\)  is the policy model being trained,  \\( \pi_{\mathrm{ref}} \\)  is the reference model,  \\( \sigma \\)  is the sigmoid function, and  \\( \beta > 0 \\)  is a hyperparameter that controls the strength of the preference signal.

## Logged metrics - HERE

While training and evaluating we record the following reward metrics:

* `global_step`: The total number of optimizer steps taken so far.
* `epoch`: The current epoch number, based on dataset iteration.
* `num_tokens`: The total number of tokens processed so far.
* `loss`: The average cross-entropy loss computed over non-masked tokens in the current logging interval.
* `entropy`: The average entropy of the model's predicted token distribution over non-masked tokens.
* `mean_token_accuracy`: The proportion of non-masked tokens for which the model’s top-1 prediction matches the ground truth token.
* `learning_rate`: The current learning rate, which may change dynamically if a scheduler is used.
* `grad_norm`: The L2 norm of the gradients, computed before gradient clipping.

## Customization

### Model initialization

You can directly pass the kwargs of the [`~transformers.AutoModelForCausalLM.from_pretrained()`] method to the [`DPOConfig`]. For example, if you want to load a model in a different precision, analogous to

```python
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16)
```

you can do so by passing the `model_init_kwargs={"dtype": torch.bfloat16}` argument to the [`DPOConfig`].

```python
from trl import DPOConfig

training_args = DPOConfig(
    model_init_kwargs={"dtype": torch.bfloat16},
)
```

Note that all keyword arguments of [`~transformers.AutoModelForCausalLM.from_pretrained()`] are supported.

### Packing

[`DPOTrainer`] supports _example packing_, where multiple examples are packed in the same input sequence to increase training efficiency. To enable packing, simply pass `packing=True` to the [`DPOConfig`] constructor.

```python
training_args = DPOConfig(packing=True)
```

For more details on packing, see [Packing](reducing_memory_usage#packing).

### Train on assistant messages only

To train on assistant messages only, use a [conversational](dataset_formats#conversational) dataset and set `assistant_only_loss=True` in the [`DPOConfig`]. This setting ensures that loss is computed **only** on the assistant responses, ignoring user or system messages.

```python
training_args = DPOConfig(assistant_only_loss=True)
```

![train_on_assistant](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/train_on_assistant.png)

> [!WARNING]
> This functionality is only available for chat templates that support returning the assistant tokens mask via the `&#123;% generation %&#125;` and `&#123;% endgeneration %&#125;` keywords. For an example of such a template, see [HugggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/main/chat_template.jinja#L76-L82).

### Train on completion only

To train on completion only, use a [prompt-completion](dataset_formats#prompt-completion) dataset. By default, the trainer computes the loss on the completion tokens only, ignoring the prompt tokens. If you want to train on the full sequence, set `completion_only_loss=False` in the [`DPOConfig`].

![train_on_completion](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/train_on_completion.png)

> [!TIP]
> Training on completion only is compatible with training on assistant messages only. In this case, use a [conversational](dataset_formats#conversational) [prompt-completion](dataset_formats#prompt-completion) dataset and set `assistant_only_loss=True` in the [`DPOConfig`].

### Train adapters with PEFT

We support tight integration with 🤗 PEFT library, allowing any user to conveniently train adapters and share them on the Hub, rather than training the entire model.

```python
from datasets import load_dataset
from trl import DPOTrainer
from peft import LoraConfig

dataset = load_dataset("trl-lib/Capybara", split="train")

trainer = DPOTrainer(
    "Qwen/Qwen3-0.6B",
    train_dataset=dataset,
    peft_config=LoraConfig()
)

trainer.train()
```

You can also continue training your [`~peft.PeftModel`]. For that, first load a `PeftModel` outside [`DPOTrainer`] and pass it directly to the trainer without the `peft_config` argument being passed.

```python
from datasets import load_dataset
from trl import DPOTrainer
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained("trl-lib/Qwen3-4B-LoRA", is_trainable=True)
dataset = load_dataset("trl-lib/Capybara", split="train")

trainer = DPOTrainer(
    model=model,
    train_dataset=dataset,
)

trainer.train()
```

> [!TIP]
> When training adapters, you typically use a higher learning rate (≈1e‑4) since only new parameters are being learned.
>
> ```python
> DPOConfig(learning_rate=1e-4, ...)
> ```

### Train with Liger Kernel

Liger Kernel is a collection of Triton kernels for LLM training that boosts multi-GPU throughput by 20%, cuts memory use by 60% (enabling up to 4× longer context), and works seamlessly with tools like FlashAttention, PyTorch FSDP, and DeepSpeed. For more information, see [Liger Kernel Integration](liger_kernel_integration).

### Rapid Experimentation for DPO

RapidFire AI is an open-source experimentation engine that sits on top of TRL and lets you launch multiple DPO configurations at once, even on a single GPU. Instead of trying configurations sequentially, RapidFire lets you **see all their learning curves earlier, stop underperforming runs, and clone promising ones with new settings in flight** without restarting. For more information, see [RapidFire AI Integration](rapidfire_integration).

### Train with Unsloth

Unsloth is an open‑source framework for fine‑tuning and reinforcement learning that trains LLMs (like Llama, Mistral, Gemma, DeepSeek, and more) up to 2× faster with up to 70% less VRAM, while providing a streamlined, Hugging Face–compatible workflow for training, evaluation, and deployment. For more information, see [Unsloth Integration](unsloth_integration).

## Instruction tuning example

**Instruction tuning** teaches a base language model to follow user instructions and engage in conversations. This requires:

1. **Chat template**: Defines how to structure conversations into text sequences, including role markers (user/assistant), special tokens, and turn boundaries. Read more about chat templates in [Chat templates](https://huggingface.co/docs/transformers/chat_templating#templates).
2. **Conversational dataset**: Contains instruction-response pairs

This example shows how to transform the [Qwen 3 0.6B Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) model into an instruction-following model using the [Capybara dataset](https://huggingface.co/datasets/trl-lib/Capybara) and a chat template from [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B). The DPO Trainer automatically handles tokenizer updates and special token configuration.

```python
from trl import DPOConfig, DPOTrainer
from datasets import load_dataset

trainer = DPOTrainer(
    model="Qwen/Qwen3-0.6B-Base",
    args=DPOConfig(
        output_dir="Qwen3-0.6B-Instruct",
        chat_template_path="HuggingFaceTB/SmolLM3-3B",
    ),
    train_dataset=load_dataset("trl-lib/Capybara", split="train"),
)
trainer.train()
```

> [!WARNING]
> Some base models, like those from Qwen, have a predefined chat template in the model's tokenizer. In these cases, it is not necessary to apply [`clone_chat_template()`], as the tokenizer already handles the formatting. However, it is necessary to align the EOS token with the chat template to ensure the model's responses terminate correctly. In these cases, specify `eos_token` in [`DPOConfig`]; for example, for `Qwen/Qwen2.5-1.5B`, one should set `eos_token="<|im_end|>"`.

Once trained, your model can now follow instructions and engage in conversations using its new chat template.

```python
>>> from transformers import pipeline
>>> pipe = pipeline("text-generation", model="Qwen3-0.6B-Instruct/checkpoint-5000")
>>> prompt = "<|im_start|>user\nWhat is the capital of France? Answer in one word.<|im_end|>\n<|im_start|>assistant\n"
>>> response = pipe(prompt)
>>> response[0]["generated_text"]
'<|im_start|>user\nWhat is the capital of France? Answer in one word.<|im_end|>\n<|im_start|>assistant\nThe capital of France is Paris.'
```

Alternatively, use the structured conversation format (recommended):

```python
>>> prompt = [{"role": "user", "content": "What is the capital of France? Answer in one word."}]
>>> response = pipe(prompt)
>>> response[0]["generated_text"]
[{'role': 'user', 'content': 'What is the capital of France? Answer in one word.'}, {'role': 'assistant', 'content': 'The capital of France is Paris.'}]
```

## Tool Calling with DPO

The [`DPOTrainer`] fully supports fine-tuning models with _tool calling_ capabilities. In this case, each dataset example should include:

* The conversation messages, including any tool calls (`tool_calls`) and tool responses (`tool` role messages)
* The list of available tools in the `tools` column, typically provided as JSON schemas

For details on the expected dataset structure, see the [Dataset Format — Tool Calling](dataset_formats#tool-calling) section.

## Training Vision Language Models

[`DPOTrainer`] fully supports training Vision-Language Models (VLMs). To train a VLM, you need to provide a dataset with an additional `images` column containing the images to be processed. For more information on the expected dataset structure, see the [Dataset Format — Vision Dataset](dataset_formats#vision-dataset) section.
An example of such a dataset is the [LLaVA Instruct Mix](https://huggingface.co/datasets/trl-lib/llava-instruct-mix).

```python
from trl import DPOConfig, DPOTrainer
from datasets import load_dataset

trainer = DPOTrainer(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    args=DPOConfig(max_length=None),
    train_dataset=load_dataset("trl-lib/llava-instruct-mix", split="train"),
)
trainer.train()
```

> [!TIP]
> For VLMs, truncating may remove image tokens, leading to errors during training. To avoid this, set `max_length=None` in the [`DPOConfig`]. This allows the model to process the full sequence length without truncating image tokens.
>
> ```python
> DPOConfig(max_length=None, ...)
> ```
>
> Only use `max_length` when you've verified that truncation won't remove image tokens for the entire dataset.

## DPOTrainer

[[autodoc]] experimental.dpo.DPOTrainer
    - train
    - save_model
    - push_to_hub

## DPOConfig

[[autodoc]] experimental.dpo.DPOConfig
