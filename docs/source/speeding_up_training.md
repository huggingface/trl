# Speeding Up Training

<Tip warning={true}>

Section under construction. Feel free to contribute!

</Tip>

## vLLM for fast generation in online methods

Online methods such as GRPO or Online DPO require the model to generate completions, which is often a slow process and can significantly impact training time.
To speed up generation, you can use [vLLM](https://github.com/vllm-project/vllm), a library that enables fast generation through, among other things, PagedAttention. TRL's online trainers support vLLM, greatly improving training speed.

To use [vLLM](https://github.com/vllm-project/vllm), first install it using:

```bash
pip install vllm
```

or 

```bash
pip install "trl[vllm]"
```

<hfoptions id="vllm examples">
<hfoption id="Online DPO">

Then, enable it by passing `use_vllm=True` in the training arguments.

```python
from trl import OnlineDPOConfig

training_args = OnlineDPOConfig(..., use_vllm=True)
```

</hfoption>
<hfoption id="GRPO">

Then, enable it by passing `use_vllm=True` in the training arguments.

```python
from trl import GRPOConfig

training_args = GRPOConfig(..., use_vllm=True)
```

The strategy here is to use a dedicated GPU for generation powered by vLLM, while using the remainder for training.

<Tip warning={true}>

When using vLLM, an additional GPU is required exclusively for generation. This means you need at least two available GPUs and must ensure that one remains unused by the trainer. To achieve this, run the training with `--num_processes <NUMBER_OF_GPUs - 1>`.

For example, if you have 4 GPUs, set `--num_processes 3` to allocate three GPUs for training while reserving one for generation.
```bash
accelerate launch --multi_gpu --num_processes 3 train_grpo.py
```

![](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/1_gpu_for_generation.png)

</Tip>

You can further tune the vLLM configuration by setting a specific `vllm_device` and `vllm_gpu_memory_utilization` in the [`GRPOConfig`].

```python
training_args = GRPOConfig(
    ...,
    use_vllm=True,
    vllm_device="cuda:4",
    vllm_gpu_memory_utilization=0.7,
)
```

</hfoption>
</hfoptions>
