# Speeding Up Training

This guide covers various methods to accelerate training in TRL. Each technique includes minimal examples with links to more comprehensive documentation.

## vLLM for fast generation in online methods

[Online methods](index#online-methods) such as GRPO or Online DPO require the model to generate completions, which is often a slow process and can significantly impact training time.
To speed up generation, you can use [vLLM](https://github.com/vllm-project/vllm), a library that enables fast generation through, among other things, PagedAttention. TRL's online trainers support vLLM, greatly improving training speed. For more details, see [vLLM Integration](vllm_integration).

To use [vLLM](https://github.com/vllm-project/vllm), first install it using:

```bash
pip install trl[vllm]
```

<hfoptions id="vllm examples">
<hfoption id="Online DPO">

First, start a vLLM server by running:

```bash
trl vllm-serve --model <model_name>
```

Then, run the training script and pass `use_vllm=True` in the training arguments.

```python
from trl.experimental.online_dpo import OnlineDPOConfig

training_args = OnlineDPOConfig(..., use_vllm=True)
```

</hfoption>
<hfoption id="GRPO">

First, start a vLLM server by running:

```bash
trl vllm-serve --model <model_name>
```

Then, run the training script and pass `use_vllm=True` in the training arguments.

```python
from trl import GRPOConfig

training_args = GRPOConfig(..., use_vllm=True)
```

You can customize the server configuration by passing additional arguments. For more information, see [vLLM integration](vllm_integration).

> [!WARNING]
> When using vLLM, ensure that the GPUs assigned for training and generation are separate to avoid resource conflicts. For instance, if you plan to use 4 GPUs for training and another 4 for vLLM generation, you can specify GPU allocation using `CUDA_VISIBLE_DEVICES`.  
>
> Set GPUs **0-3** for vLLM generation:  
>
> ```sh
> CUDA_VISIBLE_DEVICES=0,1,2,3 trl vllm-serve --model <model_name>
> ```  
>
> And GPUs **4-7** for training:
>
> ```sh
> CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch train.py
> ```

</hfoption>
<hfoption id="RLOO">

First, start a vLLM server by running:

```bash
trl vllm-serve --model <model_name>
```

Then, run the training script and pass `use_vllm=True` in the training arguments.

```python
from trl import RLOOConfig

training_args = RLOOConfig(..., use_vllm=True)
```

You can customize the server configuration by passing additional arguments. For more information, see [vLLM integration](vllm_integration).

> [!WARNING]
> When using vLLM, ensure that the GPUs assigned for training and generation are separate to avoid resource conflicts. For instance, if you plan to use 4 GPUs for training and another 4 for vLLM generation, you can specify GPU allocation using `CUDA_VISIBLE_DEVICES`.  
>
> Set GPUs **0-3** for vLLM generation:
>
> ```sh
> CUDA_VISIBLE_DEVICES=0,1,2,3 trl vllm-serve --model <model_name>
> ```  
>
> And GPUs **4-7** for training:
>
> ```sh
> CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch train.py
> ```

</hfoption>
</hfoptions>

## Optimized attention implementations

TRL supports various optimized attention implementations that can significantly speed up training while reducing memory usage. You can use either a pre-optimized kernels directly from the [Kernels Hub](kernels_hub) or a manually built attention backend.

<hfoptions id="attention examples">
<hfoption id="Kernels from Hub">

You can use pre-optimized attention kernels from the Hub without manual compilation:

```python
from trl import SFTConfig

training_args = SFTConfig(..., model_init_kwargs={"attn_implementation": "kernels-community/flash-attn2"})
```

Other options include `kernels-community/vllm-flash-attn3` and `kernels-community/paged-attention`.

Optimized attention works across all TRL trainers. For more details, see [Kernels Hub Integration](kernels_hub).

</hfoption>
<hfoption id="Manual build">

> [!WARNING]
> Manually building optimized attention backends is complex and time-consuming. It's never recommended unless absolutely necessary. Consider using Kernels from the Hub instead, as described in the previous section.

If you have manually installed an optimized attention backend like Flash Attention 2, you can specify it in the training arguments:

```python
from trl import SFTConfig

training_args = SFTConfig(..., model_init_kwargs={"attn_implementation": "flash_attention_2"})
```

</hfoption>
</hfoptions>

## Liger Kernel for memory optimization

Liger Kernel is a collection of Triton kernels designed for LLM training that can increase throughput by 20% and reduce memory usage by 60%.

<hfoptions id="liger">
<hfoption id="SFT">

```python
from trl import SFTConfig

training_args = SFTConfig(..., use_liger_kernel=True)
```

</hfoption>
<hfoption id="DPO">

```python
from trl import DPOConfig

training_args = DPOConfig(..., use_liger_kernel=True)
```

</hfoption>
<hfoption id="GRPO">

```python
from trl import GRPOConfig

training_args = GRPOConfig(..., use_liger_kernel=True)
```

</hfoption>
<hfoption id="KTO">

```python
from trl.experimental.kto import KTOConfig

training_args = KTOConfig(..., use_liger_kernel=True)
```

</hfoption>
<hfoption id="GKD">

```python
from trl.experimental.gkd import GKDConfig

training_args = GKDConfig(..., use_liger_kernel=True)
```

</hfoption>
</hfoptions>

For more information, see [Liger Kernel Integration](liger_kernel_integration).

## Mixed precision training

Mixed precision training using bf16 or fp16 can speed up training and reduce memory usage with minimal impact on model quality.

```python
from trl import SFTConfig

training_args = SFTConfig(..., bf16=True)  # or fp16=True for older GPUs
```

Use `bf16=True` for Ampere GPUs (A100, RTX 30xx) or newer, and `fp16=True` for older GPUs. Mixed precision training is supported across all TRL trainers.
