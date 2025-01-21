# Reducing Memory Usage

<Tip warning={true}>

Section under construction. Feel free to contribute!

</Tip>

## Truncation

Sequence lengths in the dataset can vary widely. When data is batched, sequences are padded to match the longest one in the batch, which can cause high memory usage, even if most sequences are relatively short.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/why_you_should_truncate.png" alt="Truncation prompt completion" width="600"/>
</div>

To reduce memory usage, it’s important to truncate sequences to a reasonable length. While TRL trainers truncate sequences by default, you may want to adjust the default truncation length to better align with your specific use case.

<hfoptions id="dpo">
<hfoption id="DPO">

DPO truncation is applied first to the prompt and to the completion via the `max_prompt_length` and `max_completion_length` parameters. The `max_length` parameter is then used to truncate the resulting sequence.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/truncation_prompt_completion.png" alt="Truncation prompt completion" width="600"/>
</div>

To set the truncation parameters, use the following code snippet:

```python
from trl import DPOConfig

training_args = DPOConfig(..., max_prompt_length=..., max_length=...)
```

You can also use the `max_completion_length` parameter to truncate the completion, though this is less common since the goal is typically to preserve the completion's full length whenever possible.

```python
from trl import DPOConfig

training_args = DPOConfig(..., max_completion_length=...)
```

</hfoption>
<hfoption id="SFT">

SFT truncation is applied to the input sequence via the `max_seq_length` parameter.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/truncation_input_ids.png" alt="Truncation input ids" width="600"/>
</div>

To set the truncation parameter, use the following code snippet:

```python
from trl import SFTConfig

training_args = SFTConfig(..., max_seq_length=...)
```

</hfoption>
</hfoptions>

## Packing

<Tip>

This technique applies only to SFT.

</Tip>


[Truncation](#truncation) has several drawbacks:
1. **Loss of information**: Key data at the end of a sequence may be discarded.
2. **Choosing truncation length**: Too short loses data; too long undermines efficiency.

Packing, introduced in [Raffel et al., 2020](https://huggingface.co/papers/1910.10683), addresses these issues by grouping sequences instead of truncating. It concatenates and splits dataset sequences into the desired lengths.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/packing.png" alt="Packing" width="600"/>
</div>

Packing eliminates padding, preserves all sequence information, and allows for flexible sequence lengths, making it a more efficient alternative to truncation. To enable packing, use `packing=True` in the [`SFTConfig`]:

```python
from trl import SFTConfig

training_args = SFTConfig(..., packing=True, max_seq_length=512)
```

<Tip warning={true}>

Packing may cause batch contamination, where adjacent sequences influence one another. This can be problematic for some applications. For more details, see [#1230](https://github.com/huggingface/trl/issues/1230).

</Tip>

## Disabling model gathering for generation in online methods

When using DeepSpeed ZeRO-3, model weights are sharded across multiple GPUs. Online methods involve generating completions from the model as part of the training process. During this step, the model weights are temporarily gathered on a single GPU for generation. For very large models, this gathering can lead to out-of-memory (OOM) errors, as described in this issue: [#2250](https://github.com/huggingface/trl/issues/2250#issue-2598304204).

If you encounter this issue, you can disable the gathering of model weights for generation by setting the following parameter:

<hfoptions id="ds3_gather_for_generation">
<hfoption id="Online DPO">

```python
from trl import OnlineDPOConfig

training_args = OnlineDPOConfig(..., ds3_gather_for_generation=False)
```

</hfoption>
<hfoption id="PPO">

```python
from trl import PPOConfig

training_args = PPOConfig(..., ds3_gather_for_generation=False)
```

</hfoption>
<hfoption id="RLOO">

```python
from trl import RLOOConfig

training_args = RLOOConfig(..., ds3_gather_for_generation=False)
```

</hfoption>
</hfoptions>

This adjustment prevents model weights from being gathered, avoiding OOM errors, but it may result in slower generation speeds.
