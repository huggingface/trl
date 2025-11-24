# vLLM Integration

This document will guide you through the process of using vLLM with TRL for faster generation in online methods like GRPO and Online DPO. We first summarize a tl;dr on how to use vLLM with TRL, and then we will go into the details of how it works under the hood.

> [!WARNING]
> TRL currently only supports vLLM version `0.10.2`. Please ensure you have this version installed to avoid compatibility issues.

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
CUDA_VISIBLE_DEVICES=0,1,2,3 trl vllm-serve --model Qwen/Qwen2.5-7B --tensor-parallel-size 2 --data-parallel-size 2
```

Once the server is running, you can use it to generate completions for training. In the example below, we are using the different supported trainers using the vLLM server for generation. The `--tensor-parallel-size` and `--data-parallel-size` arguments control how the model and data are sharded across GPUs.

In this example, we are sharding two copies of the model across 4 GPUs. Increasing data parallelism increases throughput, while increasing tensor parallelism allows for serving larger models. Then, run the training script on different GPUs (e.g., GPUs 4-7) by passing `use_vllm=True` in the training arguments as follows:

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
    args=GRPOConfig(use_vllm=True),
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
    args=OnlineDPOConfig(use_vllm=True),
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
    args=NashMDConfig(use_vllm=True),
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
)

trainer.train()
```

</hfoption>
<hfoption id="XPO">

```python
from datasets import load_dataset
from trl import XPOTrainer, XPOConfig
from trl.rewards import accuracy_reward

dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

trainer = XPOTrainer(
    model="Qwen/Qwen2.5-7B",
    args=XPOConfig(use_vllm=True),
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
    args=RLOOConfig(use_vllm=True),
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

### 🎬 Flashback: Why do we need to use vLLM in online methods?

Online methods like GRPO or Online DPO require the model to generate completions during training, which are then used to compute reward signals. However, generation can be extremely time-consuming, especially with large or reasoning models. In the default setup (without vLLM), completions are generated using the [(unwrapped) model's `generate` method](https://github.com/huggingface/trl/blob/f3e8c2304428ef16e9ae5de9e5741ed84d533b7b/trl/trainer/grpo_trainer.py#L965C39-L965C66). This approach quickly becomes a major bottleneck — generation is slow and inefficient, particularly for large batches or models. As a result, training times increase significantly, and overall efficiency drops. To address this, we turn to vLLM, which enables much faster and more scalable generation, helping eliminate this bottleneck in online methods.

### 🤔 How does vLLM solve the slow generation issue?

If you've ever done autoregressive decoder training, you know all the input tokens to the LLM produce their attention key and value tensors, and these tensors are kept in GPU memory to later generate subsequent tokens based on them. These cached key and value tensors are often referred to as the KV cache. However, storing the KV cache occupies a lot of memory, so vLLM uses a technique called **PagedAttention** to solve this problem. PagedAttention, which is inspired by the OS’s virtual memory concept, stores continuous keys and values in **non-contiguous memory space**, which is much more efficient. The details of this are beyond the scope of this document, but in short, it allows the model to store the keys and values in a more efficient way, reducing the memory footprint and speeding up the generation process. If you are interested, make sure to check out the [vLLM PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) for more details.

## How vLLM Works (Under the Hood) 🔍

### 🤔 What exactly happens when you run `trl vllm-serve --model <model_name>`?

When you run for example

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 trl vllm-serve --model Qwen/Qwen2.5-7B --tensor-parallel-size 1 --data-parallel-size 4
```

the following happens:

![vllm](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/vllm-doc.png)

1. vLLM first spawns multiple workers to handle incoming requests in parallel. The number of workers is determined by multiplying the `--tensor-parallel-size` and `--data-parallel-size` values. In this example, it spawns 4 workers (1 × 4).
Each worker operates independently and processes a chunk of the incoming requests — which are basically the prompts sent to the server for generation. A key point to understand is that these 4 workers are running in parallel, and each one is responsible for handling a subset of the total incoming load.

2. Once the incoming requests (prompts) are distributed across the workers, the model starts generating completions. Internally, the model’s weights are split across multiple GPUs based on the `--tensor-parallel-size` argument — this is how tensor parallelism is handled. Meanwhile, data parallelism (controlled by `--data-parallel-size`) ensures that different sets of requests are processed independently across the workers. In short: tensor parallelism splits the model across GPUs, and data parallelism splits the batch of requests across different model replicas.

3. Although the GPUs process requests independently and in parallel, they still need to communicate with each other. Remember that each GPU handles only a slice of the incoming prompts (for example, with 4 GPUs and 8 prompts using `--data-parallel-size=4`, each GPU processes 2 prompts).
This GPU-to-GPU communication is managed efficiently by NVIDIA’s NCCL library. The communication mainly ensures that each GPU gets its correct portion of the incoming requests — it’s lightweight and doesn’t interfere with generation itself.
Separately, the number of completions to generate per prompt is controlled by the `num_generations` setting in the GRPO config. For instance, if you set `num_generations=2` (like in the picture above), each prompt will have 2 completions. So, with 8 prompts and `num_generations=2`, you would end up with 16 completions total — regardless of the number of GPUs or parallelism settings.

### 🥸 More detail on what happens under the hood when running the server

- The vLLM server starts by running the command: `trl vllm-serve --model Qwen/Qwen2.5-7B`.
- Once the server is running, it generates completions based on requests from the client (trainer) using `vllm_client.generate` [these lines](https://github.com/huggingface/trl/blob/cc044e35b285be7dc062764b3364e1e684db4c7c/trl/trainer/grpo_trainer.py#L1025-L1035).
- The client (trainer) then requests these completions from the server.
- These completions are used to compute the reward signal.
- Based on the reward signal and the model’s output, the loss is computed, and the backward pass is performed to update the model’s weights.
- **Note**: The server only handles completion generation — it doesn’t train the model. Therefore, the model’s weights aren’t updated on the server. Once the backward pass is complete, the client sends the updated weights to the server using `vllm_client.update_named_param(name, param.data)`.

When using vLLM, ensure the GPUs assigned for training and generation are separate to avoid NCCL communication conflicts. If you do not set the `CUDA_VISIBLE_DEVICES` environment variable, the training script will use all available GPUs by default, which may lead to device conflicts. Starting from TRL next release after v0.19.1, the code automatically detects and prevents same-device usage, raising a error at the vllm server process:

```log
RuntimeError: Attempting to use the same CUDA device for multiple distinct roles/ranks within the same communicator. 
Ensure that trainer is using different devices than vLLM server.
```

For example, if you want to use GPUs 4–7 for training while the server runs on GPUs 0-3, set:

```sh
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch train.py
```

## Advanced usage

### 🍷 More customization options with vLLM?

You can customize the server configuration by passing additional arguments.

```txt
$ trl vllm-serve --help
usage: trl vllm-serve [-h] --model MODEL [--revision REVISION] [--tensor_parallel_size TENSOR_PARALLEL_SIZE] [--data_parallel_size DATA_PARALLEL_SIZE] [--host HOST]
                      [--port PORT] [--gpu_memory_utilization GPU_MEMORY_UTILIZATION] [--dtype DTYPE] [--max_model_len MAX_MODEL_LEN]
                      [--enable_prefix_caching ENABLE_PREFIX_CACHING] [--enforce_eager [ENFORCE_EAGER]] [--kv_cache_dtype KV_CACHE_DTYPE]
                      [--trust_remote_code [TRUST_REMOTE_CODE]] [--log_level LOG_LEVEL] [--vllm_model_impl VLLM_MODEL_IMPL]

options:
  -h, --help            show this help message and exit
  --model MODEL         Model name or path to load the model from. (default: None)
  --revision REVISION   Revision to use for the model. If not specified, the default branch will be used. (default: None)
  --tensor_parallel_size TENSOR_PARALLEL_SIZE, --tensor-parallel-size TENSOR_PARALLEL_SIZE
                        Number of tensor parallel workers to use. (default: 1)
  --data_parallel_size DATA_PARALLEL_SIZE, --data-parallel-size DATA_PARALLEL_SIZE
                        Number of data parallel workers to use. (default: 1)
  --host HOST           Host address to run the server on. (default: 0.0.0.0)
  --port PORT           Port to run the server on. (default: 8000)
  --gpu_memory_utilization GPU_MEMORY_UTILIZATION, --gpu-memory-utilization GPU_MEMORY_UTILIZATION
                        Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache on the device dedicated to generation
                        powered by vLLM. Higher values will increase the KV cache size and thus improve the model's throughput. However, if the value is too high,
                        it may cause out-of-memory (OOM) errors during initialization. (default: 0.9)
  --dtype DTYPE         Data type to use for vLLM generation. If set to 'auto', the data type will be automatically determined based on the model configuration.
                        Find the supported values in the vLLM documentation. (default: auto)
  --max_model_len MAX_MODEL_LEN, --max-model-len MAX_MODEL_LEN
                        If set, the `max_model_len` to use for vLLM. This can be useful when running with reduced `vllm_gpu_memory_utilization`, leading to a
                        reduced KV cache size. If not set, vLLM will use the model context size, which might be much larger than the KV cache, leading to
                        inefficiencies. (default: None)
  --enable_prefix_caching ENABLE_PREFIX_CACHING, --enable-prefix-caching ENABLE_PREFIX_CACHING
                        Whether to enable prefix caching in vLLM. If set to `True`, ensure that the model and the hardware support this feature. (default: None)
  --enforce_eager [ENFORCE_EAGER], --enforce-eager [ENFORCE_EAGER]
                        Whether to enforce eager execution. If set to `True`, we will disable CUDA graph and always execute the model in eager mode. If `False`
                        (default behavior), we will use CUDA graph and eager execution in hybrid. (default: False)
  --kv_cache_dtype KV_CACHE_DTYPE, --kv-cache-dtype KV_CACHE_DTYPE
                        Data type to use for KV cache. If set to 'auto', the dtype will default to the model data type. (default: auto)
  --trust_remote_code [TRUST_REMOTE_CODE], --trust-remote-code [TRUST_REMOTE_CODE]
                        Whether to trust remote code when loading models. Set to True to allow executing code from model repositories. This is required for some
                        custom models but introduces security risks. (default: False)
  --log_level LOG_LEVEL, --log-level LOG_LEVEL
                        Log level for uvicorn. Possible choices: 'critical', 'error', 'warning', 'info', 'debug', 'trace'. (default: info)
  --vllm_model_impl VLLM_MODEL_IMPL, --vllm-model-impl VLLM_MODEL_IMPL
                        Model implementation to use for vLLM. Must be one of `transformers` or `vllm`. `transformers`: Use the `transformers` backend for model
                        implementation. `vllm`: Use the `vllm` library for model implementation. (default: vllm)
```

### 💆🏻‍♀️ What's the best distributed setup?

![tp dp throughput 8 gpus](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/tp_dp_throughput_8_gpus.png)
![tp dp throughput 4 gpus](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/tp_dp_throughput_4_gpus.png)

First and foremost, always remember that the optimal setup depends on:

- The model size
- The number of GPUs you have
- The GPU memory size
- The batch size you are using
- The number of requests you are sending to the server (prompts)
- The `max_model_len` you are using (this is the max length of the input sequence that the model can process, a.k.a. the context window size)
- The number of completions you are generating for each request (`num_generations`)

Given these factors, our experiments on the Qwen model family (3B, 7B, 14B, 32B) using 8 H100 GPUs show that:

- For reasonable-sized models (3B–14B) and a moderate context window (`max_len < 8k`), using full capacity for data parallelism gives better throughput. The setup `(tp=1, dp=8)` yields the best results.
- For larger models (32B) and longer context windows (`max_len > 8k`), a smaller DP size combined with some model-side parallelism performs better. For example, `(tp=2, dp=4)` is a good setup for 32B models with a larger context window.

### vLLM with Transformers Backend

vLLM can use the **Transformers backend** for model implementations, which works for both LLMs and VLMs.
To enable this, set `vllm_model_impl="transformers"` in your configuration or pass it via the command-line argument.

For more details, check out [vLLM Transformers Backend](https://blog.vllm.ai/2025/04/11/transformers-backend.html).

Example:

```sh
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen
2.5-VL-3B-Instruct --tensor-parallel-size 1 --port 8000 --enforce_eager --vllm_model_impl transformers
```

### Modes of Using vLLM During Training

TRL supports **two modes** for integrating vLLM during training: **server mode** and **colocate mode**.

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
    vllm_mode="server",  # default value, can be omitted
)
```

</hfoption>
<hfoption id="OnlineDPO">

```python
from trl.experimental.online_dpo import OnlineDPOConfig

training_args = OnlineDPOConfig(
    ...,
    use_vllm=True,
    vllm_mode="server",  # default value, can be omitted
)
```

</hfoption>
<hfoption id="NashMD">

```python
from trl.experimental.nash_md import NashMDConfig

training_args = NashMDConfig(
    ...,
    use_vllm=True,
    vllm_mode="server",  # default value, can be omitted
)
```

</hfoption>
<hfoption id="XPO">

```python
from trl.experimental.xpo import XPOConfig

training_args = XPOConfig(
    ...,
    use_vllm=True,
    vllm_mode="server",  # default value, can be omitted
)
```

</hfoption>
<hfoption id="RLOO">

```python
from trl import RLOOConfig

training_args = RLOOConfig(
    ...,
    use_vllm=True,
    vllm_mode="server",  # default value, can be omitted
)
```

</hfoption>
</hfoptions>

#### Colocate Mode

In **colocate mode**, vLLM runs inside the trainer process and shares GPU memory with the training model.
This avoids launching a separate server and can improve GPU utilization, but may lead to memory contention on the training GPUs.

Example configuration:

<hfoptions id="vllm examples">
<hfoption id="GRPO">

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    ...,
    use_vllm=True,
    vllm_mode="colocate",
)
```

</hfoption>
<hfoption id="OnlineDPO">

```python
from trl.experimental.online_dpo import OnlineDPOConfig

training_args = OnlineDPOConfig(
    ...,
    use_vllm=True,
    vllm_mode="colocate",
)
```

</hfoption>
<hfoption id="NashMD">

```python
from trl.experimental.nash_md import NashMDConfig

training_args = NashMDConfig(
    ...,
    use_vllm=True,
    vllm_mode="colocate",
)
```

</hfoption>
<hfoption id="XPO">

```python
from trl.experimental.xpo import XPOConfig

training_args = XPOConfig(
    ...,
    use_vllm=True,
    vllm_mode="colocate",
)
```

</hfoption>
<hfoption id="RLOO">

```python
from trl import RLOOConfig

training_args = RLOOConfig(
    ...,
    use_vllm=True,
    vllm_mode="colocate",
)
```

</hfoption>
</hfoptions>

> [!WARNING]
> Check the documentation of the trainer you are using for specific details on vLLM usage and parameters.

> [!WARNING]
> To reduce GPU memory usage when running vLLM, consider [enabling vLLM sleep mode](reducing_memory_usage#vllm-sleep-mode).
