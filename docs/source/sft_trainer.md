# SFT Trainer

[![](https://img.shields.io/badge/All_models-SFT-blue)](https://huggingface.co/models?other=sft,trl) [![](https://img.shields.io/badge/smol_course-Chapter_1-yellow)](https://github.com/huggingface/smol-course/tree/main/1_instruction_tuning)

## Overview

TRL supports the Supervised Fine-Tuning (SFT) Trainer for training language models.

This post-training method was contributed by [Younes Belkada](https://huggingface.co/ybelkada).

## Quick start

This example demonstrates how to train a language model using the [`SFTTrainer`] from TRL. We train a [Qwen 3 0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) model on the [Capybara dataset](https://huggingface.co/datasets/trl-lib/Capybara), a compact, diverse multi-turn dataset to benchmark reasoning and generalization.

```python
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

trainer = SFTTrainer(
    model="Qwen/Qwen3-0.6B",
    train_dataset=load_dataset("trl-lib/Capybara", split="train"),
)
trainer.train()
```

<iframe src="https://trl-lib-trackio.hf.space/?project=trl-documentation&metrics=train/loss,train/mean_token_accuracy,train/num_tokens&sidebar=hidden" style="width: 100%; min-width: 300px; max-width: 800px;" height="830" frameBorder="0"/>

## Expected dataset type and format

SFT supports both [language modeling](dataset_formats#language-modeling) and [prompt-completion](dataset_formats#prompt-completion) datasets. The [`SFTTrainer`] is compatible with both [standard](dataset_formats#standard) and [conversational](dataset_formats#conversational) dataset formats. When provided with a conversational dataset, the trainer will automatically apply the chat template to the dataset.

```python
# Standard language modeling
{"text": "The sky is blue."}

# Conversational language modeling
{"messages": [{"role": "user", "content": "What color is the sky?"},
              {"role": "assistant", "content": "It is blue."}]}

# Standard prompt-completion
{"prompt": "The sky is",
 "completion": " blue."}

# Conversational prompt-completion
{"prompt": [{"role": "user", "content": "What color is the sky?"}],
 "completion": [{"role": "assistant", "content": "It is blue."}]}
```

If your dataset is not in one of these formats, you can preprocess it to convert it into the expected format. Here is an example with the [FreedomIntelligence/medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) dataset:

```python
from datasets import load_dataset

dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en")

def preprocess_function(example):
    return {
        "prompt": [{"role": "user", "content": example["Question"]}],
        "completion": [
            {"role": "assistant", "content": f"<think>{example['Complex_CoT']}</think>{example['Response']}"}
        ],
    }

dataset = dataset.map(preprocess_function, remove_columns=["Question", "Response", "Complex_CoT"])
print(next(iter(dataset["train"])))
```

```json
{
    "prompt": [
        {
            "content": "Given the symptoms of sudden weakness in the left arm and leg, recent long-distance travel, and the presence of swollen and tender right lower leg, what specific cardiac abnormality is most likely to be found upon further evaluation that could explain these findings?",
            "role": "user",
        }
    ],
    "completion": [
        {
            "content": "<think>Okay, let's see what's going on here. We've got sudden weakness [...] clicks into place!</think>The specific cardiac abnormality most likely to be found in [...] the presence of a PFO facilitating a paradoxical embolism.",
            "role": "assistant",
        }
    ],
}
```

## Looking deeper into the SFT method

Supervised Fine-Tuning (SFT) is the simplest and most commonly used method to adapt a language model to a target dataset. The model is trained in a fully supervised fashion using pairs of input and output sequences. The goal is to minimize the negative log-likelihood (NLL) of the target sequence, conditioning on the input.

This section breaks down how SFT works in practice, covering the key steps: **preprocessing**, **tokenization** and **loss computation**.

### Preprocessing and tokenization

During training, each example is expected to contain a **text field** or a **(prompt, completion)** pair, depending on the dataset format. For more details on the expected formats, see [Dataset formats](dataset_formats).
The `SFTTrainer` tokenizes each input using the model's tokenizer. If both prompt and completion are provided separately, they are concatenated before tokenization.

### Computing the loss

![](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/sft_figure.png)

The loss used in SFT is the **token-level cross-entropy loss**, defined as:

$$
\mathcal{L}_{\text{SFT}}(\theta) = - \sum_{t=1}^{T} \log p_\theta(y_t \mid y_{<t}),
$$
  
where  \\( y_t \\) is the target token at timestep  \\( t \\), and the model is trained to predict the next token given the previous ones. In practice, padding tokens are masked out during loss computation.

### Label shifting and masking

During training, the loss is computed using a **one-token shift**: the model is trained to predict each token in the sequence based on all previous tokens. Specifically, the input sequence is shifted right by one position to form the target labels.
Padding tokens (if present) are ignored in the loss computation by applying an ignore index (default: `-100`) to the corresponding positions. This ensures that the loss focuses only on meaningful, non-padding tokens.

## Logged metrics

* `global_step`: The total number of optimizer steps taken so far.
* `epoch`: The current epoch number, based on dataset iteration.
* `num_tokens`: The total number of tokens processed so far.
* `loss`: The average cross-entropy loss computed over non-masked tokens in the current logging interval.
* `mean_token_accuracy`: The proportion of non-masked tokens for which the model’s top-1 prediction matches the ground truth token.
* `learning_rate`: The current learning rate, which may change dynamically if a scheduler is used.
* `grad_norm`: The L2 norm of the gradients, computed before gradient clipping.

## Customization

### Model initialization

You can directly pass the kwargs of the [`~transformers.AutoModelForCausalLM.from_pretrained()`] method to the [`SFTConfig`]. For example, if you want to load a model in a different precision, analogous to

```python
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16)
```

you can do so by passing the `model_init_kwargs={"torch_dtype": torch.bfloat16}` argument to the [`SFTConfig`].

```python
from trl import SFTConfig

training_args = SFTConfig(
    model_init_kwargs={"torch_dtype": torch.bfloat16},
)
```

Note that all keyword arguments of `from_pretrained()` are supported.

### Packing

[`SFTTrainer`] supports _example packing_, where multiple examples are packed in the same input sequence to increase training efficiency. To enable packing, simply pass `packing=True` to the [`SFTConfig`] constructor.

```python
training_args = SFTConfig(packing=True)
```

For more defails on packing, see [Packing](reducing_memory_usage#packing).

### Train on assistant messages only

To train on assistant messages only, use a [conversational](dataset_formats#conversational) dataset and set `assistant_only_loss=True` in the [`SFTConfig`]. This setting ensures that loss is computed **only** on the assistant responses, ignoring user and system and user messages.

```python
training_args = SFTConfig(assistant_only_loss=True)
```

![](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/train_on_assistant.png)

> [!WARNING]
> This functionality is only available for chat templates that support returning the assistant tokens mask via the `{% raw %}{% generation %}{% endraw %}` keyword. For an example of such an template, see [Qwen/Qwen3-8B/discussions/14](https://huggingface.co/Qwen/Qwen3-8B/discussions/14).

### Train on completion only

To train on completion only, use a [prompt-completion](dataset_formats#prompt-completion) dataset. By default, the trainer computes the loss on the completion tokens only, ignoring the prompt tokens. If you want to train on the full sequence, set `completion_only_loss=False` in the [`SFTConfig`].

![](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/train_on_completion.png)

<Tip>
Training on completion only is compatible with training on assistant messages only. In this case, use a [conversational](dataset_formats#conversational) [prompt-completion](dataset_formats#prompt-completion) dataset and set `assistant_only_loss=True` in the [`SFTConfig`].
</Tip>

### Train with Liger Kernel

Liger Kernel is a collection of Triton kernels for LLM training that boosts multi-GPU throughput by 20%, cuts memory use by 60% (enabling up to 4× longer context), and works seamlessly with tools like Flash Attention, PyTorch FSDP, and DeepSpeed. For more information, see [Liger Kernel Integration](liger_kernel_integration).

### Train with Unsloth

Unsloth is an open‑source framework for fine‑tuning and reinforcement learning that trains LLMs (like Llama, Mistral, Gemma, DeepSeek, and more) up to 2× faster with up to 70% less VRAM, while providing a streamlined, Hugging Face–compatible workflow for training, evaluation, and deployment. For more information, see [Unsloth Integration](unsloth_integration).

## Instruction tuning example

**Instruction tuning** teaches a base language model to follow user instructions and engage in conversations. This requires:

1. **Chat template**: Defines how to structure conversations into text sequences, including role markers (user/assistant), special tokens, and turn boundaries. Read more about chat templates in [Chat templates](https://huggingface.co/docs/transformers/chat_templating#templates).
2. **Conversational dataset**: Contains instruction-response pairs

This example shows how to transform the [Qwen 3 0.6B Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) model into an instruction-following model using the [Capybara dataset](https://huggingface.co/datasets/trl-lib/Capybara) and a chat template from [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B). The SFT Trainer automatically handles tokenizer updates and special token configuration.

```python
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

trainer = SFTTrainer(
    model="Qwen/Qwen3-0.6B-Base",
    args=SFTConfig(
        output_dir="Qwen3-0.6B-Instruct",
        chat_template_path="HuggingFaceTB/SmolLM3-3B",
    ),
    train_dataset=load_dataset("trl-lib/Capybara", split="train"),
)
trainer.train()
```

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

## Tool Calling with SFT

The SFT trainer fully supports fine-tuning models with *tool calling* capabilities. In this case, each dataset example should include:

* The conversation messages, including any tool calls (`tool_calls`) and tool responses (`tool` role messages)
* The list of available tools in the `tools` column, typically provided as JSON schemas

For details on the expected dataset structure, see the [Dataset Format — Tool Calling](dataset_formats#tool-calling) section.

## SFTTrainer

[[autodoc]] SFTTrainer

## SFTConfig

[[autodoc]] SFTConfig
