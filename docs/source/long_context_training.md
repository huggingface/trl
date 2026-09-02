# Training Beyond 1M Tokens

An agent is capable of working on a task for hours: reading files, running commands, and carrying every turn of the session forward. That accumulated session runs to hundreds of thousands of tokens, which is why frontier models now advertise million-token context windows.

Making a model good at that much context means training it on sequences that long, and that is where it gets difficult: a million-token sequence does not fit on a GPU, and it does not fit on eight of them either, not without help.

The example in this guide trains on exactly one such sequence per step, on a single 8-GPU node.

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
- **380 s per step.** One step is one sequence, so that is a little over six minutes per sequence.

Naively, this sequence needs 288 GB per GPU. The rest of this guide is how it gets to 56.

## What just happened

Everything below builds up from a setup we already know: one GPU, a few thousand tokens per sequence. Things start breaking as the sequence grows, and they break in a fixed order.

### The loss is the first thing to break

Take Qwen3-4B on a single 80 GB card and grow the sequence. The first thing to run out of memory is not the model. It is the loss. Check the memory profile.

<img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/long_context_profile_plain.png" alt="Memory through one training step, with a large block of memory between the forward and backward passes"/>

The peak is between the forward and backward passes, which is where the loss is computed. To understand why the peak is there, we need to look at what the last stages of decoding do and how the loss is built from them.

<img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/long_context_plain_loss.png" alt="Hidden states times the lm_head weights giving a logits matrix of sequence by vocabulary, then the per-token loss"/>

The last layer of the decoder outputs a hidden state of shape \\( L \times H \\), where \\( L \\) is the sequence length and \\( H \\) is the hidden size. This hidden state is then multiplied by the language modeling head weights to produce a logits matrix of shape \\( L \times V \\), where \\( V \\) is the vocabulary size. This matrix is huge, because the sequence is long and the vocabulary is large (over a hundred thousand entries). The loss is then computed with a cross entropy function from that matrix and the target labels, and materializing it is precisely what the peak is.

Can we compute the loss without materializing the whole logits matrix? Yes, we can proceed in chunks.

### Chunking the loss

To cut the head off this peak, we can compute the loss in chunks. Instead of computing the logits for the whole sequence at once, we take a few hundred rows at a time, compute the loss for those, and sum the pieces. This way, we never have to hold the entire logits matrix in memory at once.

<img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/long_context_chunked_loss.png" alt="The same computation one chunk of rows at a time, each chunk freed before the next, summing into the same loss"/>

Instead of one big multiplication, we do several smaller ones. The memory saving is large enough that the loss will never be the peak again.

The new profile looks like this.

<img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/long_context_profile_chunked.png" alt="The same step with the chunked loss, where the memory stays flat throughout"/>

The peak is gone.

How to enable this in TRL? There is nothing to do: the chunked loss is the default. If you want to opt in to the plain loss, set `loss_type="nll"` in the [`SFTConfig`].

How far does this let us push the sequence? In memory terms, this one change alone takes us from 32k tokens to 160k.

<img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/long_context_scaling.png" alt="Peak memory against sequence length, the plain loss running out of memory far earlier than the chunked loss"/>


### The model has never seen position 100,000

We can fit 160k tokens in memory now. But can the model actually read them?

Qwen3-4B was trained on sequences of 40,960 tokens. The config says so:

```python
>>> from transformers import AutoConfig
>>> AutoConfig.from_pretrained("Qwen/Qwen3-4B").max_position_embeddings
40960
```

Every position beyond that is a position the model has never encountered. Nothing crashes, but the run just trains on tokens the model cannot place. Here is the loss on every token of a 160k-token sequence, averaged over eight of them. Each dot is one token:

<img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/long_context_rope_pertoken_plain.png" alt="Per-token loss along a 160k-token sequence, flat for most of it then climbing steeply near the end"/>

The model does not fall over at 40,960. It keeps going for a while, and only starts to lose the thread past 50k or so. From there it climbs steadily, ending three times worse than it started.

<img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/long_context_rope_scaling.png" alt="The trained range of positions, the much longer range we ask for, and the same range rescaled to fit inside the trained one"/>

Each token comes with its position, and the model learned what positions between 0 and 40,960 mean. Ask it about position 100,000 and it has nothing to go on.

So instead of asking, we rescale: position 100,000 becomes position 25,000, and every position in the sequence lands back inside the range the model knows. The tokens do not change, only the position each one is given. This is what [YaRN](https://huggingface.co/papers/2309.00071) does, and the factor is simply how much further we are going: 40,960 trained, 163,840 wanted, so 4.

How to enable this in TRL? Pass the RoPE configuration when the model is loaded:

```python
training_args = SFTConfig(
    ...,
    model_init_kwargs={
        "rope_parameters": {
            "rope_type": "yarn",
            "rope_theta": 1_000_000,  # has to match what the model ships with
            "factor": 4.0,
            "original_max_position_embeddings": 40960,
        },
    },
)
```

Does it work? The same measurement, with the rescaled run on top:

<img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/long_context_rope_pertoken.png" alt="The same per-token loss with a second run using rescaled positions, which stays flat where the first one climbs"/>

The rescaled run stays flat all the way to 160k. Over the last 20k tokens it averages 2.8 against 7.3 without the rescaling.

Look at the dots rather than the lines and you can see what the average is made of. Early in the sequence both runs have the same shape: most tokens cheap, a few expensive. Late in the unscaled run the whole cloud has lifted, so it is not a handful of hard tokens dragging the mean up, it is the model reading worse everywhere.
