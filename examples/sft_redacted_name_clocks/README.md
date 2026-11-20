# Teach REDACTED_NAME to read analog clocks

Reading an analog clock is easy for people and hard for vision-language models. This example measures how well [REDACTED_NAME](https://huggingface.co/REDACTED_HUB_ID) reads clocks, fine-tunes it with LoRA on one H100, and measures again.

## 1. Can REDACTED_NAME read a clock?

The test split of [migonsa/analog_clocks_combinations_for_finetuning](https://huggingface.co/datasets/migonsa/analog_clocks_combinations_for_finetuning) holds 25,920 clock images, each with the time it shows. [`evaluate.py`](evaluate.py) asks the model the time on a shuffled sample of them and checks its answer three ways: to the second, to the minute, and the hour alone.

```sh
python examples/sft_redacted_name_clocks/evaluate.py REDACTED_HUB_ID --reasoning_effort low --max_new_tokens 1024 --num_clocks 32
```

```
REDACTED_HUB_ID: 0.0% to the second, 15.6% to the minute, 50.0% the hour
```

REDACTED_NAME reasons before it answers, even at the lowest reasoning effort, and on a clock it writes 600 to 1,000 tokens: which hand is the hour hand, where each one points, what time that makes. Hence the low effort and the small sample. The reasoning is sensible, the readings are approximate: it gets the hour half the time, the minute about one time in six, and the second never.

## 2. The dataset

The training split holds 103,680 synthetic clocks in three styles: plain (`base`), stretched and skewed (`distorted`), and with unusual hands (`modified_hands`). Here is one of them:

```python
>>> from datasets import load_dataset
>>> dataset = load_dataset("migonsa/analog_clocks_combinations_for_finetuning", split="train")
>>> dataset[0]
{'type': 'distorted', 'question': 'What time is shown on the clock in the given image?', 'answer': '09:01:02', 'image': <PIL.PngImagePlugin.PngImageFile image mode=RGBA size=343x482>}
```

[`sft_redacted_name_clocks.py`](sft_redacted_name_clocks.py) turns each row into a conversation with a `map`: the user sends the image and the question, the model answers the time.

```python
def to_messages(example):
    return {
        "messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": example["question"]}]},
            {"role": "assistant", "content": [{"type": "text", "text": example["answer"]}]},
        ],
        "images": [example["image"]],
        # Answer directly, without reasoning first
        "chat_template_kwargs": {"reasoning_effort": "none"},
    }
```

The answers carry no reasoning, so the model learns to answer right away. The `chat_template_kwargs` column renders each conversation with the reasoning effort set to `"none"`, the setting the fine-tuned model is then used with.

## 3. Fine-tune

```sh
python examples/sft_redacted_name_clocks/sft_redacted_name_clocks.py
```

[`sft_redacted_name_clocks.py`](sft_redacted_name_clocks.py) trains a LoRA adapter (rank 16, on every linear layer) on 10,000 clocks for one epoch: 1,250 steps of 8 clocks, 53 minutes on one H100, with a peak of 55 GiB. Two settings matter for this model:

- `model_init_kwargs={"dtype": "bfloat16"}`: TRL loads a model id in float32 unless told otherwise, and the model then doesn't fit on one GPU.
- `assistant_only_loss=True`: the loss is on the answer only, not on the image and the question.

Token accuracy passes 0.9 within 200 steps:

![Token accuracy during training](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/sft_redacted_name_clocks/token_accuracy.png)

## 4. Measure again

```sh
python examples/sft_redacted_name_clocks/evaluate.py REDACTED_HUB_ID --adapter REDACTED_FILE_NAME-clocks
python examples/sft_redacted_name_clocks/evaluate.py REDACTED_HUB_ID --adapter REDACTED_FILE_NAME-clocks --num_clocks 32
```

```
REDACTED_FILE_NAME-clocks: 93.5% to the second, 94.5% to the minute, 98.0% the hour
REDACTED_FILE_NAME-clocks: 93.8% to the second, 93.8% to the minute, 96.9% the hour
```

The first line is on 200 clocks, the second on the 32 the original model was measured on:

| Model | To the second | To the minute | The hour | Tokens per answer |
| --- | --- | --- | --- | --- |
| REDACTED_NAME, low reasoning effort | 0.0% | 15.6% | 50.0% | 600 to 1,000 |
| Fine-tuned | 93.8% | 93.8% | 96.9% | 9 |

The fine-tuned model reads the time to the second on more than 9 clocks in 10, and answers in 9 tokens instead of reasoning for a thousand. Here are three test clocks, one of each style, with the answers of both models:

<table>
  <tr>
    <td><img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/sft_redacted_name_clocks/clock-base.png" width="200" alt="A plain clock showing 07:03:06"></td>
    <td><img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/sft_redacted_name_clocks/clock-distorted.png" width="200" alt="A distorted clock showing 06:54:13"></td>
    <td><img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/sft_redacted_name_clocks/clock-modified_hands.png" width="200" alt="A clock with arrow hands showing 04:34:42"></td>
  </tr>
  <tr>
    <td>07:03:06. Original: "approximately <b>7:05</b>" after 604 tokens. Fine-tuned: 07:03:06.</td>
    <td>06:54:13. Original: "approximately <b>10:35</b>" after 777 tokens. Fine-tuned: 06:54:13.</td>
    <td>04:34:42. Original: "about <b>4:35</b>" after 954 tokens. Fine-tuned: 04:34:42.</td>
  </tr>
</table>

The clocks are synthetic: the fine-tuned model reads these three styles, and nothing here says how well it reads a photo of a real clock.
