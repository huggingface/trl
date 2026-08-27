# vLLM Integration

This document will guide you through the process of using vLLM with TRL for faster generation in online methods like GRPO and Online DPO. We first summarize a tl;dr on how to use vLLM with TRL, and then we will go into the details of how it works under the hood.

> [!WARNING]
> TRL currently only supports vLLM versions from `0.19.0` to `0.27.1`. Please ensure you have a version in this range installed to avoid compatibility issues.

> [!TIP]
> The following trainers currently support generation with vLLM:
>
> - [`GRPOTrainer`]
> - [`RLOOTrainer`]
> - [`experimental.nash_md.NashMDTrainer`]
> - [`experimental.online_dpo.OnlineDPOTrainer`]
> - [`experimental.xpo.XPOTrainer`]

## 🚀 How can I use vLLM with TRL to speed up training?

💡 **Note**: Resources required for this specific example: a single node with 8 GPUs.

> [!WARNING]
> When using vLLM with TRL, the **vLLM server** and the **trainer** must run on **separate CUDA devices** to prevent conflicts.
> For guidance on configuring this properly, see [Modes of using vLLM during training](#modes-of-using-vllm-during-training).

First, install vLLM using the following command:

```bash
pip install "trl[vllm]"
```

Then run the server on specific GPUs (e.g., GPUs 0-3):

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen2.5-7B --tensor-parallel-size 4 \
    --weight-transfer-config '{"backend": "nccl"}' \
    --logprobs-mode processed_logprobs \
    --max-logprobs -1
```

Once the server is running, you can use it to generate completions for training. In the example below, we are using the different supported trainers using the vLLM server for generation. The `--tensor-parallel-size` and `--data-parallel-size` arguments control how the model and data are sharded across GPUs.

In this example, we shard one model across 4 GPUs with tensor parallelism. Then, run the training script on different GPUs (e.g., GPUs 4-7) by passing `use_vllm=True` in the training arguments as follows:

Sample of a simple `train.py` script:

<hfoptions id="vllm examples">
<hfoption id="GRPO">

```python
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from trl.rewards import accuracy_reward

dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-7B",
    args=GRPOConfig(use_vllm=True, vllm_mode="server"),
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
)

trainer.train()
```

</hfoption>
<hfoption id="OnlineDPO">

```python
from datasets import load_dataset
from trl.experimental.online_dpo import OnlineDPOConfig, OnlineDPOTrainer
from trl.rewards import accuracy_reward

dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

trainer = OnlineDPOTrainer(
    model="Qwen/Qwen2.5-7B",
    args=OnlineDPOConfig(use_vllm=True, vllm_mode="server"),
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
)

trainer.train()
```

</hfoption>
<hfoption id="NashMD">

```python
from datasets import load_dataset
from trl.experimental.nash_md import NashMDConfig, NashMDTrainer
from trl.rewards import accuracy_reward

dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

trainer = NashMDTrainer(
    model="Qwen/Qwen2.5-7B",
    args=NashMDConfig(use_vllm=True, vllm_mode="server"),
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
)

trainer.train()
```

</hfoption>
<hfoption id="XPO">

```python
from datasets import load_dataset
from trl.experimental.xpo import XPOTrainer, XPOConfig
from trl.rewards import accuracy_reward

dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

trainer = XPOTrainer(
    model="Qwen/Qwen2.5-7B",
    args=XPOConfig(use_vllm=True, vllm_mode="server"),
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
)

trainer.train()
```

</hfoption>
<hfoption id="RLOO">

```python
from datasets import load_dataset
from trl import RLOOTrainer, RLOOConfig
from trl.rewards import accuracy_reward

dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

trainer = RLOOTrainer(
    model="Qwen/Qwen2.5-7B",
    args=RLOOConfig(use_vllm=True, vllm_mode="server"),
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
)

trainer.train()
```

</hfoption>
</hfoptions>

And the train command on separate GPUs from the server:

```sh
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch train.py
```

## Why using vLLM?

Online methods generate completions during training, and generating them with the model's own `generate` is the
bottleneck. vLLM serves those completions far faster, thanks to techniques like
[PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html).

## How TRL uses the server 🔍

The trainer asks for completions on the OpenAI-compatible `/v1/completions` endpoint, sending the prompt token IDs.
Multimodal prompts take a different route: the server processes the images on their own, and the resulting features
are paired with the same token IDs on `/inference/v1/generate`, since no OpenAI-compatible endpoint takes token IDs
and images at once.

The server only generates. After each optimizer step the trainer streams the updated weights into it over NCCL,
announcing them with `/start_weight_update` and `/update_weights` and committing them with `/finish_weight_update`.

## Advanced usage

### 🍷 More customization options with vLLM?

You can customize the server configuration by passing any `vllm serve` argument, for instance `--tensor-parallel-size`, `--data-parallel-size`, `--max-model-len`, `--enable-prefix-caching`, `--enforce-eager`, `--kv-cache-dtype` or `--trust-remote-code`. Run `vllm serve --help` for the full list.

Only the following are required by TRL:

| Setting | Why |
| --- | --- |
| `VLLM_SERVER_DEV_MODE=1` | Exposes the weight-transfer and prefix-cache endpoints used to push the training weights into the server. It also exposes vLLM's other development endpoints, so keep the server on a trusted network. |
| `--weight-transfer-config '{"backend": "nccl"}'` | Enables the NCCL weight-transfer engine. Use `"ipc"` instead when the trainer and the server share a GPU. |
| `--logprobs-mode processed_logprobs` | Returns logprobs after temperature scaling and logit processing, which is what the importance sampling correction expects. |
| `--max-logprobs -1` | Lifts the OpenAI-compatible cap of 20 logprobs per token, required to request the top-k teacher distribution for distillation. |

> [!WARNING]
> `trl vllm-serve` is deprecated: it now only builds this command and runs vLLM's server. It prints the exact `vllm serve` command it runs, so you can copy it and drop the wrapper.

### 💆🏻‍♀️ What's the best distributed setup?

Scale generation with `--tensor-parallel-size`. Data parallelism no longer helps dense models: since
[vLLM PR #30739](https://github.com/vllm-project/vllm/pull/30739) (released in `0.14.0`), offline data parallel
scaling for non-MoE models is not supported.

### vLLM with Transformers Backend

vLLM can use the **Transformers backend** for model implementations, which works for both LLMs and VLMs.
To enable this, set `vllm_model_impl="transformers"` in your configuration or pass it via the command-line argument.

For more details, check out [vLLM Transformers Backend](https://blog.vllm.ai/2025/04/11/transformers-backend.html).

Example:

```sh
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
    --tensor-parallel-size 1 --port 8000 --enforce-eager --model-impl transformers \
    --weight-transfer-config '{"backend": "nccl"}' \
    --logprobs-mode processed_logprobs \
    --max-logprobs -1
```

### Modes of Using vLLM During Training

TRL supports **two modes** for integrating vLLM during training: **colocate mode** (default) and **server mode**.

#### Colocate Mode

In **colocate mode**, vLLM runs inside the trainer process and shares GPU memory with the training model.
This avoids launching a separate server and can improve GPU utilization, but may lead to memory contention on the training GPUs. This is the default mode.

Example configuration:

<hfoptions id="vllm examples">
<hfoption id="GRPO">

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    ...,
    use_vllm=True,  # vllm_mode="colocate" by default
)
```

</hfoption>
<hfoption id="OnlineDPO">

```python
from trl.experimental.online_dpo import OnlineDPOConfig

training_args = OnlineDPOConfig(
    ...,
    use_vllm=True,  # vllm_mode="colocate" by default
)
```

</hfoption>
<hfoption id="NashMD">

```python
from trl.experimental.nash_md import NashMDConfig

training_args = NashMDConfig(
    ...,
    use_vllm=True,  # vllm_mode="colocate" by default
)
```

</hfoption>
<hfoption id="XPO">

```python
from trl.experimental.xpo import XPOConfig

training_args = XPOConfig(
    ...,
    use_vllm=True,  # vllm_mode="colocate" by default
)
```

</hfoption>
<hfoption id="RLOO">

```python
from trl import RLOOConfig

training_args = RLOOConfig(
    ...,
    use_vllm=True,  # vllm_mode="colocate" by default
)
```

</hfoption>
</hfoptions>

#### Server Mode

In **server mode**, vLLM runs as a separate process on dedicated GPUs and communicates with the trainer via HTTP.
This setup is ideal if you have GPUs dedicated to inference.

Example configuration:

<hfoptions id="vllm examples">
<hfoption id="GRPO">

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    ...,
    use_vllm=True,
    vllm_mode="server",
)
```

</hfoption>
<hfoption id="OnlineDPO">

```python
from trl.experimental.online_dpo import OnlineDPOConfig

training_args = OnlineDPOConfig(
    ...,
    use_vllm=True,
    vllm_mode="server",
)
```

</hfoption>
<hfoption id="NashMD">

```python
from trl.experimental.nash_md import NashMDConfig

training_args = NashMDConfig(
    ...,
    use_vllm=True,
    vllm_mode="server",
)
```

</hfoption>
<hfoption id="XPO">

```python
from trl.experimental.xpo import XPOConfig

training_args = XPOConfig(
    ...,
    use_vllm=True,
    vllm_mode="server",
)
```

</hfoption>
<hfoption id="RLOO">

```python
from trl import RLOOConfig

training_args = RLOOConfig(
    ...,
    use_vllm=True,
    vllm_mode="server",
)
```

</hfoption>
</hfoptions>

> [!WARNING]
> Check the documentation of the trainer you are using for specific details on vLLM usage and parameters.

> [!WARNING]
> To reduce GPU memory usage when running vLLM, consider [enabling vLLM sleep mode](reducing_memory_usage#vllm-sleep-mode).
