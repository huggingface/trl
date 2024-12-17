# Reducing Memory Usage

<Tip warning={true}>

Section under construction. Feel free to contribute!

</Tip>

## Truncation

Sequence lengths in the dataset can vary widely, and by default, TRL does not modify the data. When data is batched, sequences are padded to match the longest one in the batch, which can cause high memory usage, even if most sequences are relatively short.

To reduce memory usage, it’s important to truncate sequences to a reasonable length. Even discarding just a few tokens from the dataset can result in significant memory savings by minimizing unnecessary padding. Truncation is a good practice and should always be applied to ensure efficient use of resources. While the truncation limit doesn’t need to be overly restrictive, setting a sensible value is essential for optimal performance.

<hfoptions id="dpo">
<hfoption id="DPO">
Markdown for DPO
</hfoption>
<hfoption id="SFT">
Markdown for SFT
</hfoption>
something else
</hfoptions>