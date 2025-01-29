# Dataset formats and types

This guide provides an overview of the dataset formats and types supported by each trainer in TRL.

## Overview of the dataset formats and types

- The *format* of a dataset refers to how the data is structured, typically categorized as either *standard* or *conversational*.
- The *type* is associated with the specific task the dataset is designed for, such as *prompt-only* or *preference*. Each type is characterized by its columns, which vary according to the task, as shown in the table.

<table>
  <tr>
    <th>Type \ Format</th>
    <th>Standard</th>
    <th>Conversational</th>
  </tr>
  <tr>
    <td>Language modeling</td>
    <td>
      <pre><code>{"text": "The sky is blue."}</code></pre>
    </td>
    <td>
      <pre><code>{"messages": [{"role": "user", "content": "What color is the sky?"},
              {"role": "assistant", "content": "It is blue."}]}</code></pre>
    </td>
  </tr>
  <tr>
    <td>Prompt-only</td>
    <td>
      <pre><code>{"prompt": "The sky is"}</code></pre>
    </td>
    <td>
      <pre><code>{"prompt": [{"role": "user", "content": "What color is the sky?"}]}</code></pre>
    </td>
  </tr>
  <tr>
    <td>Prompt-completion</td>
    <td>
      <pre><code>{"prompt": "The sky is",
 "completion": " blue."}</code></pre>
    </td>
    <td>
      <pre><code>{"prompt": [{"role": "user", "content": "What color is the sky?"}],
 "completion": [{"role": "assistant", "content": "It is blue."}]}</code></pre>
    </td>
  </tr>
  </tr>
  <tr>
    <td>Preference</td>
    <td>
      <pre><code>{"prompt": "The sky is",
 "chosen": " blue.",
 "rejected": " green."}</code></pre>
      or, with implicit prompt:
      <pre><code>{"chosen": "The sky is blue.",
 "rejected": "The sky is green."}</code></pre>
    </td>
    <td>
      <pre><code>{"prompt": [{"role": "user", "content": "What color is the sky?"}],
 "chosen": [{"role": "assistant", "content": "It is blue."}],
 "rejected": [{"role": "assistant", "content": "It is green."}]}</code></pre>
      or, with implicit prompt:
      <pre><code>{"chosen": [{"role": "user", "content": "What color is the sky?"},
              {"role": "assistant", "content": "It is blue."}],
 "rejected": [{"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "content": "It is green."}]}</code></pre>
    </td>
  </tr>
    <td>Unpaired preference</td>
    <td>
      <pre><code>{"prompt": "The sky is",
 "completion": " blue.",
 "label": True}</code></pre>
    </td>
    <td>
      <pre><code>{"prompt": [{"role": "user", "content": "What color is the sky?"}],
 "completion": [{"role": "assistant", "content": "It is green."}],
 "label": False}</code></pre>
    </td>
  </tr>
  </tr>
    <td>Stepwise supervision</td>
    <td>
      <pre><code>{"prompt": "Which number is larger, 9.8 or 9.11?",
 "completions": ["The fractional part of 9.8 is 0.8.", 
                 "The fractional part of 9.11 is 0.11.",
                 "0.11 is greater than 0.8.",
                 "Hence, 9.11 > 9.8."],
 "labels": [True, True, False, False]}</code></pre>
    </td>
    <td></td>
  </tr>
</table>

### Formats

#### Standard

The standard dataset format typically consists of plain text strings. The columns in the dataset vary depending on the task. This is the format expected by TRL trainers. Below are examples of standard dataset formats for different tasks:

```python
# Language modeling
language_modeling_example = {"text": "The sky is blue."}
# Preference
preference_example = {"prompt": "The sky is", "chosen": " blue.", "rejected": " green."}
# Unpaired preference
unpaired_preference_example = {"prompt": "The sky is", "completion": " blue.", "label": True}
```

#### Conversational

Conversational datasets are used for tasks involving dialogues or chat interactions between users and assistants. Unlike standard dataset formats, these contain sequences of messages where each message has a `role` (e.g., `"user"` or `"assistant"`) and `content` (the message text).

```python
messages = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
    {"role": "user", "content": "I'd like to show off how chat templating works!"},
]
```

Just like standard datasets, the columns in conversational datasets vary depending on the task. Below are examples of conversational dataset formats for different tasks:

```python
# Prompt-completion
prompt_completion_example = {"prompt": [{"role": "user", "content": "What color is the sky?"}],
                             "completion": [{"role": "assistant", "content": "It is blue."}]}
# Preference
preference_example = {
    "prompt": [{"role": "user", "content": "What color is the sky?"}],
    "chosen": [{"role": "assistant", "content": "It is blue."}],
    "rejected": [{"role": "assistant", "content": "It is green."}],
}
```

Conversational datasets are useful for training chat models, but must be converted into a standard format before being used with TRL trainers. This is typically done using chat templates specific to the model being used. For more information, refer to the [Working with conversational datasets in TRL](#working-with-conversational-datasets-in-trl) section.

### Types

#### Language modeling

A language modeling dataset consists of a column `"text"` (or `"messages"` for conversational datasets) containing a full sequence of text.

```python
# Standard format
language_modeling_example = {"text": "The sky is blue."}
# Conversational format
language_modeling_example = {"messages": [
    {"role": "user", "content": "What color is the sky?"},
    {"role": "assistant", "content": "It is blue."}
]}
```

#### Prompt-only

In a prompt-only dataset, only the initial prompt (the question or partial sentence) is provided under the key `"prompt"`. The training typically involves generating the completion based on this prompt, where the model learns to continue or complete the given input.

```python
# Standard format
prompt_only_example = {"prompt": "The sky is"}
# Conversational format
prompt_only_example = {"prompt": [{"role": "user", "content": "What color is the sky?"}]}
```

For examples of prompt-only datasets, refer to the [Prompt-only datasets collection](https://huggingface.co/collections/trl-lib/prompt-only-datasets-677ea25245d20252cea00368).

<Tip>

While both the prompt-only and language modeling types are similar, they differ in how the input is handled. In the prompt-only type, the prompt represents a partial input that expects the model to complete or continue, while in the language modeling type, the input is treated as a complete sentence or sequence. These two types are processed differently by TRL. Below is an example showing the difference in the output of the `apply_chat_template` function for each type:

```python
from transformers import AutoTokenizer
from trl import apply_chat_template

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

# Example for prompt-only type
prompt_only_example = {"prompt": [{"role": "user", "content": "What color is the sky?"}]}
apply_chat_template(prompt_only_example, tokenizer)
# Output: {'prompt': '<|user|>\nWhat color is the sky?<|end|>\n<|assistant|>\n'}

# Example for language modeling type
lm_example = {"messages": [{"role": "user", "content": "What color is the sky?"}]}
apply_chat_template(lm_example, tokenizer)
# Output: {'text': '<|user|>\nWhat color is the sky?<|end|>\n<|endoftext|>'}
```

- The prompt-only output includes a `'<|assistant|>\n'`, indicating the beginning of the assistant’s turn and expecting the model to generate a completion.
- In contrast, the language modeling output treats the input as a complete sequence and terminates it with `'<|endoftext|>'`, signaling the end of the text and not expecting any additional content.

</Tip>

#### Prompt-completion

A prompt-completion dataset includes a `"prompt"` and a `"completion"`.

```python
# Standard format
prompt_completion_example = {"prompt": "The sky is", "completion": " blue."}
# Conversational format
prompt_completion_example = {"prompt": [{"role": "user", "content": "What color is the sky?"}],
                             "completion": [{"role": "assistant", "content": "It is blue."}]}
```

For examples of prompt-completion datasets, refer to the [Prompt-completion datasets collection](https://huggingface.co/collections/trl-lib/prompt-completion-datasets-677ea2bb20bbb6bdccada216).

#### Preference

A preference dataset is used for tasks where the model is trained to choose between two or more possible completions to the same prompt. This dataset includes a `"prompt"`, a `"chosen"` completion, and a `"rejected"` completion. The model is trained to select the `"chosen"` response over the `"rejected"` response.
Some dataset may not include the `"prompt"` column, in which case the prompt is implicit and directly included in the `"chosen"` and `"rejected"` completions. We recommend using explicit prompts whenever possible.

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

For examples of preference datasets, refer to the [Preference datasets collection](https://huggingface.co/collections/trl-lib/preference-datasets-677e99b581018fcad9abd82c).

Some preference datasets can be found with [the tag `dpo` on Hugging Face Hub](https://huggingface.co/datasets?other=dpo). You can also explore the [librarian-bots' DPO Collections](https://huggingface.co/collections/librarian-bots/direct-preference-optimization-datasets-66964b12835f46289b6ef2fc) to identify preference datasets.

#### Unpaired preference

An unpaired preference dataset is similar to a preference dataset but instead of having `"chosen"` and `"rejected"` completions for the same prompt, it includes a single `"completion"` and a `"label"` indicating whether the completion is preferred or not.

```python
# Standard format
unpaired_preference_example = {"prompt": "The sky is", "completion": " blue.", "label": True}
# Conversational format
unpaired_preference_example = {"prompt": [{"role": "user", "content": "What color is the sky?"}],
                               "completion": [{"role": "assistant", "content": "It is blue."}],
                               "label": True}
```

For examples of unpaired preference datasets, refer to the [Unpaired preference datasets collection](https://huggingface.co/collections/trl-lib/unpaired-preference-datasets-677ea22bf5f528c125b0bcdf).

#### Stepwise supervision

A stepwise (or process) supervision dataset is similar to an [unpaired preference](#unpaired-preference) dataset but includes multiple steps of completions, each with its own label. This structure is useful for tasks that need detailed, step-by-step labeling, such as reasoning tasks. By evaluating each step separately and providing targeted labels, this approach helps identify precisely where the reasoning is correct and where errors occur, allowing for targeted feedback on each part of the reasoning process.

```python
stepwise_example = {
    "prompt": "Which number is larger, 9.8 or 9.11?",
    "completions": ["The fractional part of 9.8 is 0.8, while the fractional part of 9.11 is 0.11.", "Since 0.11 is greater than 0.8, the number 9.11 is larger than 9.8."],
    "labels": [True, False]
}
```

For examples of stepwise supervision datasets, refer to the [Stepwise supervision datasets collection](https://huggingface.co/collections/trl-lib/stepwise-supervision-datasets-677ea27fd4c5941beed7a96e).

## Which dataset type to use?

Choosing the right dataset type depends on the task you are working on and the specific requirements of the TRL trainer you are using. Below is a brief overview of the dataset types supported by each TRL trainer.

| Trainer                 | Expected dataset type                                                                                  |
| ----------------------- | ------------------------------------------------------------------------------------------------------ |
| [`BCOTrainer`]          | [Unpaired preference](#unpaired-preference)                                                            |
| [`CPOTrainer`]          | [Preference (explicit prompt recommended)](#preference)                                                |
| [`DPOTrainer`]          | [Preference (explicit prompt recommended)](#preference)                                                |
| [`GKDTrainer`]          | [Prompt-completion](#prompt-completion)                                                                |
| [`GRPOTrainer`]         | [Prompt-only](#prompt-only)                                                                            |
| [`IterativeSFTTrainer`] | [Unpaired preference](#unpaired-preference)                                                            |
| [`KTOTrainer`]          | [Unpaired preference](#unpaired-preference) or [Preference (explicit prompt recommended)](#preference) |
| [`NashMDTrainer`]       | [Prompt-only](#prompt-only)                                                                            |
| [`OnlineDPOTrainer`]    | [Prompt-only](#prompt-only)                                                                            |
| [`ORPOTrainer`]         | [Preference (explicit prompt recommended)](#preference)                                                |
| [`PPOTrainer`]          | Tokenized language modeling                                                                            |
| [`PRMTrainer`]          | [Stepwise supervision](#stepwise-supervision)                                                          |
| [`RewardTrainer`]       | [Preference (implicit prompt recommended)](#preference)                                                |
| [`SFTTrainer`]          | [Language modeling](#language-modeling)                                                                |
| [`XPOTrainer`]          | [Prompt-only](#prompt-only)                                                                            |

<Tip>

TRL trainers only support standard dataset formats, [for now](https://github.com/huggingface/trl/issues/2071). If you have a conversational dataset, you must first convert it into a standard format.
For more information on how to work with conversational datasets, refer to the [Working with conversational datasets in TRL](#working-with-conversational-datasets-in-trl) section.

</Tip>

## Working with conversational datasets in TRL

Conversational datasets are increasingly common, especially for training chat models. However, some TRL trainers don't support conversational datasets in their raw format. (For more information, see [issue #2071](https://github.com/huggingface/trl/issues/2071).) These datasets must first be converted into a standard format.
Fortunately, TRL offers tools to easily handle this conversion, which are detailed below.

### Converting a conversational dataset into a standard dataset

To convert a conversational dataset into a standard dataset, you need to _apply a chat template_ to the dataset. A chat template is a predefined structure that typically includes placeholders for user and assistant messages. This template is provided by the tokenizer of the model you use.

For detailed instructions on using chat templating, refer to the [Chat templating section in the `transformers` documentation](https://huggingface.co/docs/transformers/en/chat_templating).

In TRL, the method you apply to convert the dataset will vary depending on the task. Fortunately, TRL provides a helper function called [`apply_chat_template`] to simplify this process. Here's an example of how to use it:

```python
from transformers import AutoTokenizer
from trl import apply_chat_template

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

example = {
    "prompt": [{"role": "user", "content": "What color is the sky?"}],
    "completion": [{"role": "assistant", "content": "It is blue."}]
}

apply_chat_template(example, tokenizer)
# Output:
# {'prompt': '<|user|>\nWhat color is the sky?<|end|>\n<|assistant|>\n', 'completion': 'It is blue.<|end|>\n<|endoftext|>'}
```

Alternatively, you can use the [`~datasets.Dataset.map`] method to apply the template across an entire dataset:

```python
from datasets import Dataset
from trl import apply_chat_template

dataset_dict = {
    "prompt": [[{"role": "user", "content": "What color is the sky?"}],
               [{"role": "user", "content": "Where is the sun?"}]],
    "completion": [[{"role": "assistant", "content": "It is blue."}],
                   [{"role": "assistant", "content": "In the sky."}]]
}

dataset = Dataset.from_dict(dataset_dict)
dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})
# Output:
# {'prompt': ['<|user|>\nWhat color is the sky?<|end|>\n<|assistant|>\n',
#             '<|user|>\nWhere is the sun?<|end|>\n<|assistant|>\n'],
#  'completion': ['It is blue.<|end|>\n<|endoftext|>', 'In the sky.<|end|>\n<|endoftext|>']}
```

<Tip warning={true}>

We recommend using the [`apply_chat_template`] function instead of calling `tokenizer.apply_chat_template` directly. Handling chat templates for non-language modeling datasets can be tricky and may result in errors, such as mistakenly placing a system prompt in the middle of a conversation.
For additional examples, see [#1930 (comment)](https://github.com/huggingface/trl/pull/1930#issuecomment-2292908614). The [`apply_chat_template`] is designed to handle these intricacies and ensure the correct application of chat templates for various tasks.

</Tip>

<Tip warning={true}>

It's important to note that chat templates are model-specific. For example, if you use the chat template from [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) with the above example, you get a different output:

```python
apply_chat_template(example, AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct"))
# Output:
# {'prompt': '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat color is the sky?<|im_end|>\n<|im_start|>assistant\n',
#  'completion': 'It is blue.<|im_end|>\n'}
```

Always use the chat template associated with the model you're working with. Using the wrong template can lead to inaccurate or unexpected results.

</Tip>

## Using any dataset with TRL: preprocessing and conversion

Many datasets come in formats tailored to specific tasks, which might not be directly compatible with TRL. To use such datasets with TRL, you may need to preprocess and convert them into the required format.

To make this easier, we provide a set of [example scripts](https://github.com/huggingface/trl/tree/main/examples/datasets) that cover common dataset conversions.

### Example: UltraFeedback dataset

Let’s take the [UltraFeedback dataset](https://huggingface.co/datasets/openbmb/UltraFeedback) as an example. Here's a preview of the dataset:

<iframe
  src="https://huggingface.co/datasets/openbmb/UltraFeedback/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

As shown above, the dataset format does not match the expected structure. It’s not in a conversational format, the column names differ, and the results pertain to different models (e.g., Bard, GPT-4) and aspects (e.g., "helpfulness", "honesty").

By using the provided conversion script [`examples/datasets/ultrafeedback.py`](https://github.com/huggingface/trl/tree/main/examples/datasets/ultrafeedback.py), you can transform this dataset into an unpaired preference type, and push it to the Hub:

```sh
python examples/datasets/ultrafeedback.py --push_to_hub --repo_id trl-lib/ultrafeedback-gpt-3.5-turbo-helpfulness
```

Once converted, the dataset will look like this:

<iframe
  src="https://huggingface.co/datasets/trl-lib/ultrafeedback-gpt-3.5-turbo-helpfulness/embed/viewer/default/train?row=0"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

Now, you can use this dataset with TRL!

By adapting the provided scripts or creating your own, you can convert any dataset into a format compatible with TRL.

## Utilities for converting dataset types

This section provides example code to help you convert between different dataset types. While some conversions can be performed after applying the chat template (i.e., in the standard format), we recommend performing the conversion before applying the chat template to ensure it works consistently.

For simplicity, some of the examples below do not follow this recommendation and use the standard format. However, the conversions can be applied directly to the conversational format without modification.

| From \ To                       | Language modeling                                                       | Prompt-completion                                                       | Prompt-only                                                       | Preference with implicit prompt                           | Preference                                                | Unpaired preference                                                       | Stepwise supervision |
| ------------------------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------------------------- | -------------------- |
| Language modeling               | N/A                                                                     | N/A                                                                     | N/A                                                               | N/A                                                       | N/A                                                       | N/A                                                                       | N/A                  |
| Prompt-completion               | [🔗](#from-prompt-completion-to-language-modeling-dataset)               | N/A                                                                     | [🔗](#from-prompt-completion-to-prompt-only-dataset)               | N/A                                                       | N/A                                                       | N/A                                                                       | N/A                  |
| Prompt-only                     | N/A                                                                     | N/A                                                                     | N/A                                                               | N/A                                                       | N/A                                                       | N/A                                                                       | N/A                  |
| Preference with implicit prompt | [🔗](#from-preference-with-implicit-prompt-to-language-modeling-dataset) | [🔗](#from-preference-with-implicit-prompt-to-prompt-completion-dataset) | [🔗](#from-preference-with-implicit-prompt-to-prompt-only-dataset) | N/A                                                       | [🔗](#from-implicit-to-explicit-prompt-preference-dataset) | [🔗](#from-preference-with-implicit-prompt-to-unpaired-preference-dataset) | N/A                  |
| Preference                      | [🔗](#from-preference-to-language-modeling-dataset)                      | [🔗](#from-preference-to-prompt-completion-dataset)                      | [🔗](#from-preference-to-prompt-only-dataset)                      | [🔗](#from-explicit-to-implicit-prompt-preference-dataset) | N/A                                                       | [🔗](#from-preference-to-unpaired-preference-dataset)                      | N/A                  |
| Unpaired preference             | [🔗](#from-unpaired-preference-to-language-modeling-dataset)             | [🔗](#from-unpaired-preference-to-prompt-completion-dataset)             | [🔗](#from-unpaired-preference-to-prompt-only-dataset)             | N/A                                                       | N/A                                                       | N/A                                                                       | N/A                  |
| Stepwise supervision            | [🔗](#from-stepwise-supervision-to-language-modeling-dataset)            | [🔗](#from-stepwise-supervision-to-prompt-completion-dataset)            | [🔗](#from-stepwise-supervision-to-prompt-only-dataset)            | N/A                                                       | N/A                                                       | [🔗](#from-stepwise-supervision-to-unpaired-preference-dataset)            | N/A                  |

### From prompt-completion to language modeling dataset

To convert a prompt-completion dataset into a language modeling dataset, concatenate the prompt and the completion.

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["The sky is", "The sun is"],
    "completion": [" blue.", " in the sky."],
})

def concat_prompt_completion(example):
    return {"text": example["prompt"] + example["completion"]}

dataset = dataset.map(concat_prompt_completion, remove_columns=["prompt", "completion"])
```

```python
>>> dataset[0]
{'text': 'The sky is blue.'}
```

### From prompt-completion to prompt-only dataset

To convert a prompt-completion dataset into a prompt-only dataset, remove the completion.

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["The sky is", "The sun is"],
    "completion": [" blue.", " in the sky."],
})

dataset = dataset.remove_columns("completion")
```

```python
>>> dataset[0]
{'prompt': 'The sky is'}
```

### From preference with implicit prompt to language modeling dataset

To convert a preference with implicit prompt dataset into a language modeling dataset, remove the rejected, and rename the column `"chosen"` to `"text"`.

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "chosen": ["The sky is blue.", "The sun is in the sky."],
    "rejected": ["The sky is green.", "The sun is in the sea."],
})

dataset = dataset.rename_column("chosen", "text").remove_columns("rejected")
```

```python
>>> dataset[0]
{'text': 'The sky is blue.'}
```

### From preference with implicit prompt to prompt-completion dataset

To convert a preference dataset with implicit prompt into a prompt-completion dataset, extract the prompt with [`extract_prompt`], remove the rejected, and rename the column `"chosen"` to `"completion"`.

```python
from datasets import Dataset
from trl import extract_prompt

dataset = Dataset.from_dict({
    "chosen": [
        [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is blue."}],
        [{"role": "user", "content": "Where is the sun?"}, {"role": "assistant", "content": "In the sky."}],
    ],
    "rejected": [
        [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is green."}],
        [{"role": "user", "content": "Where is the sun?"}, {"role": "assistant", "content": "In the sea."}],
    ],
})
dataset = dataset.map(extract_prompt).remove_columns("rejected").rename_column("chosen", "completion")
```

```python
>>> dataset[0]
{'prompt': [{'role': 'user', 'content': 'What color is the sky?'}], 'completion': [{'role': 'assistant', 'content': 'It is blue.'}]}
```

### From preference with implicit prompt to prompt-only dataset

To convert a preference dataset with implicit prompt into a prompt-only dataset, extract the prompt with [`extract_prompt`], and remove the rejected and the chosen.

```python
from datasets import Dataset
from trl import extract_prompt

dataset = Dataset.from_dict({
    "chosen": [
        [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is blue."}],
        [{"role": "user", "content": "Where is the sun?"}, {"role": "assistant", "content": "In the sky."}],
    ],
    "rejected": [
        [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is green."}],
        [{"role": "user", "content": "Where is the sun?"}, {"role": "assistant", "content": "In the sea."}],
    ],
})
dataset = dataset.map(extract_prompt).remove_columns(["chosen", "rejected"])
```

```python
>>> dataset[0]
{'prompt': [{'role': 'user', 'content': 'What color is the sky?'}]}
```

### From implicit to explicit prompt preference dataset

To convert a preference dataset with implicit prompt into a preference dataset with explicit prompt, extract the prompt with [`extract_prompt`].

```python
from datasets import Dataset
from trl import extract_prompt

dataset = Dataset.from_dict({
    "chosen": [
        [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is blue."}],
        [{"role": "user", "content": "Where is the sun?"}, {"role": "assistant", "content": "In the sky."}],
    ],
    "rejected": [
        [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is green."}],
        [{"role": "user", "content": "Where is the sun?"}, {"role": "assistant", "content": "In the sea."}],
    ],
})

dataset = dataset.map(extract_prompt)
```

```python
>>> dataset[0]
{'prompt': [{'role': 'user', 'content': 'What color is the sky?'}],
 'chosen': [{'role': 'assistant', 'content': 'It is blue.'}],
 'rejected': [{'role': 'assistant', 'content': 'It is green.'}]}
```

### From preference with implicit prompt to unpaired preference dataset

To convert a preference dataset with implicit prompt into an unpaired preference dataset, extract the prompt with [`extract_prompt`], and unpair the dataset with [`unpair_preference_dataset`].

```python
from datasets import Dataset
from trl import extract_prompt, unpair_preference_dataset

dataset = Dataset.from_dict({
    "chosen": [
        [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is blue."}],
        [{"role": "user", "content": "Where is the sun?"}, {"role": "assistant", "content": "In the sky."}],
    ],
    "rejected": [
        [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is green."}],
        [{"role": "user", "content": "Where is the sun?"}, {"role": "assistant", "content": "In the sea."}],
    ],
})

dataset = dataset.map(extract_prompt)
dataset = unpair_preference_dataset(dataset)
```

```python
>>> dataset[0]
{'prompt': [{'role': 'user', 'content': 'What color is the sky?'}],
 'completion': [{'role': 'assistant', 'content': 'It is blue.'}],
 'label': True}
```

<Tip warning={true}>

Keep in mind that the `"chosen"` and `"rejected"` completions in a preference dataset can be both good or bad.
Before applying [`unpair_preference_dataset`], please ensure that all `"chosen"` completions can be labeled as good and all `"rejected"` completions as bad.
This can be ensured by checking absolute rating of each completion, e.g. from a reward model.

</Tip>

### From preference to language modeling dataset

To convert a preference dataset into a language modeling dataset, remove the rejected, concatenate the prompt and the chosen into the `"text"` column.

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["The sky is", "The sun is"],
    "chosen": [" blue.", " in the sky."],
    "rejected": [" green.", " in the sea."],
})

def concat_prompt_chosen(example):
    return {"text": example["prompt"] + example["chosen"]}

dataset = dataset.map(concat_prompt_chosen, remove_columns=["prompt", "chosen", "rejected"])
```

```python
>>> dataset[0]
{'text': 'The sky is blue.'}
```

### From preference to prompt-completion dataset

To convert a preference dataset into a prompt-completion dataset, remove the rejected, and rename the column `"chosen"` to `"completion"`.

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["The sky is", "The sun is"],
    "chosen": [" blue.", " in the sky."],
    "rejected": [" green.", " in the sea."],
})

dataset = dataset.remove_columns("rejected").rename_column("chosen", "completion")
```

```python
>>> dataset[0]
{'prompt': 'The sky is', 'completion': ' blue.'}
```

### From preference to prompt-only dataset

To convert a preference dataset into a prompt-only dataset, remove the rejected and the chosen.

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["The sky is", "The sun is"],
    "chosen": [" blue.", " in the sky."],
    "rejected": [" green.", " in the sea."],
})

dataset = dataset.remove_columns(["chosen", "rejected"])
```

```python
>>> dataset[0]
{'prompt': 'The sky is'}
```

### From explicit to implicit prompt preference dataset

To convert a preference dataset with explicit prompt into a preference dataset with implicit prompt, concatenate the prompt to both chosen and rejected, and remove the prompt.

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": [
        [{"role": "user", "content": "What color is the sky?"}],
        [{"role": "user", "content": "Where is the sun?"}],
    ],
    "chosen": [
        [{"role": "assistant", "content": "It is blue."}],
        [{"role": "assistant", "content": "In the sky."}],
    ],
    "rejected": [
        [{"role": "assistant", "content": "It is green."}],
        [{"role": "assistant", "content": "In the sea."}],
    ],
})

def concat_prompt_to_completions(example):
    return {"chosen": example["prompt"] + example["chosen"], "rejected": example["prompt"] + example["rejected"]}

dataset = dataset.map(concat_prompt_to_completions, remove_columns="prompt")
```

```python
>>> dataset[0]
{'chosen': [{'role': 'user', 'content': 'What color is the sky?'}, {'role': 'assistant', 'content': 'It is blue.'}],
 'rejected': [{'role': 'user', 'content': 'What color is the sky?'}, {'role': 'assistant', 'content': 'It is green.'}]}
```

### From preference to unpaired preference dataset

To convert dataset into an unpaired preference dataset, unpair the dataset with [`unpair_preference_dataset`].

```python
from datasets import Dataset
from trl import unpair_preference_dataset

dataset = Dataset.from_dict({
    "prompt": [
        [{"role": "user", "content": "What color is the sky?"}],
        [{"role": "user", "content": "Where is the sun?"}],
    ],
    "chosen": [
        [{"role": "assistant", "content": "It is blue."}],
        [{"role": "assistant", "content": "In the sky."}],
    ],
    "rejected": [
        [{"role": "assistant", "content": "It is green."}],
        [{"role": "assistant", "content": "In the sea."}],
    ],
})

dataset = unpair_preference_dataset(dataset)
```

```python
>>> dataset[0]
{'prompt': [{'role': 'user', 'content': 'What color is the sky?'}],
 'completion': [{'role': 'assistant', 'content': 'It is blue.'}],
 'label': True}
```

<Tip warning={true}>

Keep in mind that the `"chosen"` and `"rejected"` completions in a preference dataset can be both good or bad.
Before applying [`unpair_preference_dataset`], please ensure that all `"chosen"` completions can be labeled as good and all `"rejected"` completions as bad.
This can be ensured by checking absolute rating of each completion, e.g. from a reward model.

</Tip>

### From unpaired preference to language modeling dataset

To convert an unpaired preference dataset into a language modeling dataset, concatenate prompts with good completions into the `"text"` column, and remove the prompt, completion and label columns.

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["The sky is", "The sun is", "The sky is", "The sun is"],
    "completion": [" blue.", " in the sky.", " green.", " in the sea."],
    "label": [True, True, False, False],
})

def concatenate_prompt_completion(example):
    return {"text": example["prompt"] + example["completion"]}

dataset = dataset.filter(lambda x: x["label"]).map(concatenate_prompt_completion).remove_columns(["prompt", "completion", "label"])
```

```python
>>> dataset[0]
{'text': 'The sky is blue.'}
```

### From unpaired preference to prompt-completion dataset

To convert an unpaired preference dataset into a prompt-completion dataset, filter for good labels, then remove the label columns.

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["The sky is", "The sun is", "The sky is", "The sun is"],
    "completion": [" blue.", " in the sky.", " green.", " in the sea."],
    "label": [True, True, False, False],
})

dataset = dataset.filter(lambda x: x["label"]).remove_columns(["label"])
```

```python
>>> dataset[0]
{'prompt': 'The sky is', 'completion': ' blue.'}
```

### From unpaired preference to prompt-only dataset

To convert an unpaired preference dataset into a prompt-only dataset, remove the completion and the label columns.

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["The sky is", "The sun is", "The sky is", "The sun is"],
    "completion": [" blue.", " in the sky.", " green.", " in the sea."],
    "label": [True, True, False, False],
})

dataset = dataset.remove_columns(["completion", "label"])
```

```python
>>> dataset[0]
{'prompt': 'The sky is'}
```

### From stepwise supervision to language modeling dataset

To convert a stepwise supervision dataset into a language modeling dataset, concatenate prompts with good completions into the `"text"` column.

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["Blue light", "Water"],
    "completions": [[" scatters more in the atmosphere,", " so the sky is green."],
                   [" forms a less dense structure in ice,", " which causes it to expand when it freezes."]],
    "labels": [[True, False], [True, True]],
})

def concatenate_prompt_completions(example):
    completion = "".join(example["completions"])
    return {"text": example["prompt"] + completion}

dataset = dataset.filter(lambda x: all(x["labels"])).map(concatenate_prompt_completions, remove_columns=["prompt", "completions", "labels"])
```

```python
>>> dataset[0]
{'text': 'Blue light scatters more in the atmosphere, so the sky is green.'}
```

### From stepwise supervision to prompt completion dataset

To convert a stepwise supervision dataset into a prompt-completion dataset, join the good completions and remove the labels.

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["Blue light", "Water"],
    "completions": [[" scatters more in the atmosphere,", " so the sky is green."],
                   [" forms a less dense structure in ice,", " which causes it to expand when it freezes."]],
    "labels": [[True, False], [True, True]],
})

def join_completions(example):
    completion = "".join(example["completions"])
    return {"completion": completion}

dataset = dataset.filter(lambda x: all(x["labels"])).map(join_completions, remove_columns=["completions", "labels"])
```

```python
>>> dataset[0]
{'prompt': 'Blue light', 'completion': ' scatters more in the atmosphere, so the sky is green.'}
```

### From stepwise supervision to prompt only dataset

To convert a stepwise supervision dataset into a prompt-only dataset, remove the completions and the labels.

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["Blue light", "Water"],
    "completions": [[" scatters more in the atmosphere,", " so the sky is green."],
                   [" forms a less dense structure in ice,", " which causes it to expand when it freezes."]],
    "labels": [[True, False], [True, True]],
})

dataset = dataset.remove_columns(["completions", "labels"])
```

```python
>>> dataset[0]
{'prompt': 'Blue light'}
```

### From stepwise supervision to unpaired preference dataset

To convert a stepwise supervision dataset into an unpaired preference dataset, join the completions and merge the labels.

The method for merging the labels depends on the specific task. In this example, we use the logical AND operation. This means that if the step labels indicate the correctness of individual steps, the resulting label will reflect the correctness of the entire sequence.

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["Blue light", "Water"],
    "completions": [[" scatters more in the atmosphere,", " so the sky is green."],
                   [" forms a less dense structure in ice,", " which causes it to expand when it freezes."]],
    "labels": [[True, False], [True, True]],
})

def merge_completions_and_labels(example):
    return {"prompt": example["prompt"], "completion": "".join(example["completions"]), "label": all(example["labels"])}

dataset = dataset.map(merge_completions_and_labels, remove_columns=["completions", "labels"])
```

```python
>>> dataset[0]
{'prompt': 'Blue light', 'completion': ' scatters more in the atmosphere, so the sky is green.', 'label': False}
```

## Vision datasets

Some trainers also support fine-tuning vision-language models (VLMs) using image-text pairs. In this scenario, it's recommended to use a conversational format, as each model handles image placeholders in text differently. 

A conversational vision dataset differs from a standard conversational dataset in two key ways:

1. The dataset must contain the key `images` with the image data.
2. The `"content"` field in messages must be a list of dictionaries, where each dictionary specifies the type of data: `"image"` or `"text"`.

Example:

```python
# Textual dataset:
"content": "What color is the sky?"

# Vision dataset:
"content": [
    {"type": "image"}, 
    {"type": "text", "text": "What color is the sky in the image?"}
]
```

An example of a conversational vision dataset is the [openbmb/RLAIF-V-Dataset](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset). Below is an embedded view of the dataset's training data, allowing you to explore it directly:

<iframe
  src="https://huggingface.co/datasets/trl-lib/rlaif-v/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

