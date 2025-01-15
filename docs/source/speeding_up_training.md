# Speeding Up Training

<Tip warning={true}>

Section under construction. Feel free to contribute!

</Tip>

## vLLM for fast generation in online methods

Online methods such as Online DPO or Nash-MD require the model to generate completions, which is often a slow process and can significantly impact training time.
To speed up generation, you can use [vLLM](https://github.com/vllm-project/vllm), a library that enables fast generation through PagedAttention. TRL's online trainers support vLLM, greatly improving training speed.

To use vLLM, first install it using:
```bash
pip install vllm
```

<hfoptions id="vllm examples">
<hfoption id="Online DPO">

Then, enable it by passing `use_vllm=True` in the training arguments.

```python
from trl import OnlineDPOConfig

training_args = DPOConfig(..., use_vllm=True)
```

</hfoption>
</hfoptions>
