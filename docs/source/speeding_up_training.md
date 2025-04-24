# Speeding Up Training

<Tip warning={true}>

Section under construction. Feel free to contribute!

</Tip>

## vLLM for fast generation in GRPO

GRPO requires the model to generate completions, which is often a slow process and can significantly impact training time.
To speed up generation, you can use [vLLM](https://github.com/vllm-project/vllm), a library that enables fast generation through, among other things, PagedAttention. TRL's online trainers support vLLM, greatly improving training speed.

To use [vLLM](https://github.com/vllm-project/vllm), first install it using:

```bash
pip install vllm
```

or 

```bash
pip install "trl[vllm]"
```

First, start a vLLM server by running:

```bash
trl vllm-serve --model <model_name>
```

Then, run the training script and pass `use_vllm=True` in the training arguments.

```python
from trl import GRPOConfig

training_args = GRPOConfig(..., use_vllm=True)
```

You can customize the server configuration by passing additional arguments.

```sh
$ trl vllm-serve --help
usage: trl vllm-serve [-h] --model MODEL [--revision REVISION] [--tensor_parallel_size TENSOR_PARALLEL_SIZE]
                      [--data_parallel_size DATA_PARALLEL_SIZE] [--host HOST] [--port PORT]
                      [--gpu_memory_utilization GPU_MEMORY_UTILIZATION] [--dtype DTYPE] [--max_model_len MAX_MODEL_LEN]
                      [--enable_prefix_caching ENABLE_PREFIX_CACHING]

options:
  -h, --help            Show this help message and exit
  --model MODEL         Model name or path to load the model from. (default: None)
  --revision REVISION   Revision to use for the model. If not specified, the default branch will be used. (default:
                        None)
  --tensor_parallel_size TENSOR_PARALLEL_SIZE, --tensor-parallel-size TENSOR_PARALLEL_SIZE
                        Number of tensor parallel workers to use. (default: 1)
  --data_parallel_size DATA_PARALLEL_SIZE, --data-parallel-size DATA_PARALLEL_SIZE
                        Number of data parallel workers to use. (default: 1)
  --host HOST           Host address to run the server on. (default: 0.0.0.0)
  --port PORT           Port to run the server on. (default: 8000)
  --gpu_memory_utilization GPU_MEMORY_UTILIZATION, --gpu-memory-utilization GPU_MEMORY_UTILIZATION
                        Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV
                        cache on the device dedicated to generation powered by vLLM. Higher values will increase the KV
                        cache size and thus improve the model's throughput. However, if the value is too high, it may
                        cause out-of-memory (OOM) errors during initialization. (default: 0.9)
  --dtype DTYPE         Data type to use for vLLM generation. If set to 'auto', the data type will be automatically
                        determined based on the model configuration. Find the supported values in the vLLM
                        documentation. (default: auto)
  --max_model_len MAX_MODEL_LEN, --max-model-len MAX_MODEL_LEN
                        If set, the `max_model_len` to use for vLLM. This can be useful when running with reduced
                        `vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use
                        the model context size, which might be much larger than the KV cache, leading to
                        inefficiencies. (default: None)
  --enable_prefix_caching ENABLE_PREFIX_CACHING, --enable-prefix-caching ENABLE_PREFIX_CACHING
                        Whether to enable prefix caching in vLLM. If set to `True`, ensure that the model and the
                        hardware support this feature. (default: None)
```

<Tip warning={true}>

When using vLLM, ensure that the GPUs assigned for training and generation are separate to avoid resource conflicts. For instance, if you plan to use 4 GPUs for training and another 4 for vLLM generation, you can specify GPU allocation using `CUDA_VISIBLE_DEVICES`.  

Set GPUs **0-3** for vLLM generation:  
```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 trl vllm-serve --model <model_name>
```  

And GPUs **4-7** for training:  
```sh
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch train.py
```  

</Tip>


### Choose the optimal value TP and DP

Depending on the various parameters like the model size, the batch size, the number of GPUs or even the desired completion length, the optimal values for tensor parallelism (TP) and data parallelism (DP) can vary. We provide a simple calculator to help you choose the best values for your use case.

You'll need:
- The model size
- The maximum completion length
- The per-device batch size
- The number of GPUs
- The gradient accumulation steps
- The number of generations per prompt
- The number of GPUs for the vLLM server

#### 1. Compute the prompt batch size

The prompt batch size is the number of prompts sent to the server per request. It is computed as follows:

$$
\text{Prompt Batch Size} = \frac{\text{Per Device Batch Size} \times \text{Number of GPUs} \times \text{Gradient Accumulation Steps}}{\text{Number of Generations per Prompt}}
$$

Example:
- Per Device Batch Size: 8
- Number of GPUs: 4
- Gradient Accumulation Steps: 16
- Number of Generations per Prompt: 8

The prompt batch size is:

$$
\text{Prompt Batch Size} = \frac{8 \times 4 \times 16}{8} = 64
$$

#### 2.

Then, using the following figure, you can choose the optimal values for TP and DP.

![TP and DP values](https://raw.githubusercontent.com/vllm-project/vllm/main/docs/images/tp_dp.png)

Example:
- Model: Qwen/Qwen2.5-7B
- Max completion length: 2048

The optimal configuration is TP=1 and DP=8