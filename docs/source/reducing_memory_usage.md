# Reducing Memory Usage

<Tip warning={true}>

Section under construction. Feel free to contribute!

</Tip>

## Truncation

Sequence lengths in the dataset can vary widely, and by default, TRL does not modify the data. When data is batched, sequences are padded to match the longest one in the batch, which can cause high memory usage, even if most sequences are relatively short.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/why_you_should_truncate.png" alt="Truncation prompt completion" width="600"/>
</div>

To reduce memory usage, it’s important to truncate sequences to a reasonable length. Even discarding just a few tokens from the dataset can result in significant memory savings by minimizing unnecessary padding. Truncation is a good practice and should always be applied to ensure efficient use of resources. While the truncation limit doesn’t need to be overly restrictive, setting a sensible value is essential for optimal performance.

<hfoptions id="dpo">
<hfoption id="DPO">

DPO truncation is applied first to the prompt and to the completion via the `max_prompt_length` and `max_completion_length` parameters. The `max_length` parameter is then used to truncate the resulting sequence.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/truncation_prompt_completion.png" alt="Truncation prompt completion" width="600"/>
</div>

To set the truncation parameters, use the following code snippet:

```python
from trl import DPOConfig

training_args = DPOConfig(..., max_prompt_length=..., max_completion_length=..., max_length=...)
```

</hfoption>
<hfoption id="SFT">

SFT truncation is applied to the input sequence via the `max_length` parameter.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/truncation_input_ids.png" alt="Truncation input ids" width="600"/>
</div>

To set the truncation parameter, use the following code snippet:

```python
from trl import SFTConfig

training_args = SFTConfig(..., max_length=...)
```

</hfoption>
</hfoptions>