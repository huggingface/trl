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

TRL supports various optimized attention implementations that can significantly speed up training while reducing memory usage. You can use either locally installed backends (like Flash Attention 2) or pull pre-optimized kernels directly from the [Kernels Hub](kernels_hub).

<hfoptions id="attention examples">
<hfoption id="Flash Attention 2">

To enable Flash Attention 2, pass `attn_implementation="flash_attention_2"` in the model initialization arguments:

```python
from trl import SFTConfig

training_args = SFTConfig(..., model_init_kwargs={"attn_implementation": "flash_attention_2"})
```

</hfoption>
<hfoption id="Kernels from Hub">

You can use pre-optimized attention kernels from the Hub without manual compilation:

```python
from trl import SFTConfig

training_args = SFTConfig(..., model_init_kwargs={"attn_implementation": "kernels-community/flash-attn2"})
```

Other options include `kernels-community/vllm-flash-attn3` and `kernels-community/paged-attention`.

</hfoption>
</hfoptions>

Optimized attention works across all TRL trainers. For more details, see [Kernels Hub Integration](kernels_hub) and [Reducing Memory Usage](reducing_memory_usage#padding-free).

## PEFT for parameter-efficient training

[PEFT](https://huggingface.co/docs/peft/index) (Parameter-Efficient Fine-Tuning) methods like LoRA significantly reduce memory usage and training time by only training a small number of adapter parameters instead of the full model.

```python
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    peft_config=peft_config,
    args=training_args,
)
```

For more details, see [PEFT Integration](peft_integration).

## Liger Kernel for memory optimization

Liger Kernel is a collection of Triton kernels designed for LLM training that can increase throughput by 20% and reduce memory usage by 60%.

<hfoptions id="liger examples">
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
from trl import KTOConfig

training_args = KTOConfig(..., use_liger_kernel=True)
```

</hfoption>
</hfoptions>

For more information, see [Liger Kernel Integration](liger_kernel_integration).

## Gradient checkpointing for memory savings

Gradient checkpointing trades compute for memory by not storing all intermediate activations during the forward pass, recomputing them during the backward pass instead.

```python
from trl import SFTConfig

training_args = SFTConfig(..., gradient_checkpointing=True)
```

Gradient checkpointing is available across all TRL trainers. For more memory optimization techniques, see the [Transformers Performance Guide](https://huggingface.co/docs/transformers/perf_train_gpu_one#gradient-checkpointing).

## Mixed precision training

Mixed precision training using bf16 or fp16 can speed up training and reduce memory usage with minimal impact on model quality.

```python
from trl import SFTConfig

training_args = SFTConfig(..., bf16=True)  # or fp16=True for older GPUs
```

Use `bf16=True` for Ampere GPUs (A100, RTX 30xx) or newer, and `fp16=True` for older GPUs. Mixed precision training is supported across all TRL trainers.
