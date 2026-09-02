# Training Beyond 1M Tokens

An agent is capable of working a task for hours: reading files, running commands, and carrying every turn of the session forward. That accumulated session runs to hundreds of thousands of tokens, which is why frontier models are now quoted with million-token context windows.

Making a model good over that much context means training it on sequences that long, and that is where it gets difficult: a million-token sequence does not fit on a GPU, and it does not fit on eight of them either, not without help.

This guide trains on exactly one such sequence per step, on a single 8-GPU node.

## Run it

You need one node with 8 H100s (or better), and transformers 5.16 or later.

```sh
accelerate launch \
    --config_file examples/sft_qwen3_8b_1m_context/context_parallel_8gpu.yaml \
    examples/sft_qwen3_8b_1m_context/sft_qwen3_8b_1m_context.py
```

The script fine-tunes Qwen3-8B on books from [PG-19](https://huggingface.co/datasets/emozilla/pg19), joined end to end until each one is about a million tokens long. After ten minutes or so of loading and tokenizing, the first step lands:

```
{'loss': 4.311, 'grad_norm': 29.25, 'num_tokens': 1049000.0, 'epoch': 0.25}
  8%|▊         | 1/12 [06:20<1:09:41, 380.10s/it]
```

Three numbers are worth reading:

- **`num_tokens` is 1.049e6.** That is one sequence, not a batch of short ones. Every token attends to every earlier token, across the whole million.
- **Loss starts at 4.3**, which is a normal starting loss. A run that is subtly misconfigured for long context starts around 10 instead, and we will come back to why.
- **380 s per step.** One step is one sequence, so that is a little over six minutes per book.

Naively, this sequence needs 288 GB per GPU. The rest of this guide is how it gets to 56.

## What just happened

Everything below builds up from a setup we already know: one GPU, a few thousand tokens per sequence. Things start breaking as the sequence grows, and they break in a fixed order.

### The loss is the first thing to break

Take Qwen3-0.6B on a single 80 GB card and grow the sequence. The first thing to cause the OOM is not the model. It is the loss. Check the memory profile.

<img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/long_context_profile_plain.png" alt="Memory through one training step, with a large block of memory between the forward and backward passes"/>

The peak is between the forward and backward passes, which is where the loss is computed. To understand why the peak is there, we need to look at what the last stages of decoding do and how the loss is built from them.

<img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/long_context_plain_loss.png" alt="Hidden states times the lm_head weights giving a logits matrix of sequence by vocabulary, then the per-token loss"/>

The last layer of the decoder outputs a hidden state of shape \\( L \times H \\), where \\( L \\) is the sequence length and \\( H \\) is the hidden size. This hidden state is then multiplied by the language modeling head weights to produce a logits matrix of shape \\( L \times V \\), where \\( V \\) is the vocabulary size. This matrix is huge, because the sequence is long and the vocabulary is large (over a hundred thousand entries). The loss is then computed with a cross entropy function from this huge logits matrix and the target labels. And the peak is precisely the materialization of this huge logits matrix.
Can we compute the loss without materializing the whole logits matrix? Yes, we can proceed in chunks.

### Chunking the loss

To cut the head off this peak, we can compute the loss in chunks. Instead of computing the logits for the entire sequence at once, we can compute them for smaller segments of the sequence, calculate the loss for each segment, and then sum these losses to get the total loss. This way, we never have to hold the entire logits matrix in memory at once.

<img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/long_context_chunked_loss.png" alt="The same computation one chunk of rows at a time, each chunk freed before the next, summing into the same loss"/>

Instead of one big multiplication, we do several smaller ones. The memory saving is large enough that the loss will never be the peak again.

The new profile looks like this.

<img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/long_context_profile_chunked.png" alt="The same step with the chunked loss, where the memory stays flat throughout"/>

The peak is gone.

How to enable this in TRL? There is nothing to do, the chunked loss is the default. If you want to opt-in for the plain loss, you can set `loss_type="nll"` in the [`SFTConfig`].

How far does it allow to scale the sequence? On the same setup, we can already go to nearly 400k tokens, with this simple trick.

<img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/long_context_scaling.png" alt="Peak memory against sequence length, the plain loss failing at 48k tokens and the chunked loss reaching 512k"/>

