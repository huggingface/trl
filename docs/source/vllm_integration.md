# vLLM Integration

This document will guide you through the process of using vLLM with TRL for faster generation in online methods like GRPO and Online DPO. We first summarize a tl;dr on how to use vLLM with TRL, and then we will go into the details of how it works under the hood. Let's go! 🔥

## 🚀 How can I use vLLM with TRL to speed up training?

💡 **Note**: Resources required for this specific example: a single node with 8 GPUs.

First, install vLLM using the following command:

```bash
pip install "trl[vllm]"
```

Then run the server:

```sh
trl vllm-serve --model Qwen/Qwen2.5-7B --tensor-parallel-size 2 --data-parallel-size 2
```

Once the server is running, you can use it to generate completions for training. In the example below, we are using the `GRPOTrainer` to train a model using the vLLM server for generation. The `--tensor-parallel-size` and `--data-parallel-size` arguments control how the model and data are sharded across GPUs.

In this example, we are sharding two copies of the model across 4 GPUs. Increasing data parallelism increases throughput, while increasing tensor parallelism allows for serving larger models. Then, run the training script by passing `use_vllm=True` in the training arguments as follows:

Sample of a simple `train.py` script:

```python
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig

dataset = load_dataset("trl-lib/tldr", split="train")

# Dummy reward function: count the number of unique characters in the completions
def reward_num_unique_chars(completions, **kwargs):
    return [len(set(c)) for c in completions]

training_args = GRPOConfig(
    output_dir="my_test",
    use_vllm=True,
    bf16=True,
    gradient_checkpointing=True,
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-7B",
    args=training_args,
    reward_funcs=reward_num_unique_chars,
    train_dataset=dataset,
)

trainer.train()
```

And the train command:

```sh
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch train.py
```

## 🎬 Flashback: Why do we need to use vLLM in online methods?

Online methods like GRPO or Online DPO require the model to generate completions during training, which are then used to compute reward signals. However, generation can be extremely time-consuming, especially with large or reasoning models. In the default setup (without vLLM), completions are generated using the [(unwrapped) model's `generate` method](https://github.com/huggingface/trl/blob/f3e8c2304428ef16e9ae5de9e5741ed84d533b7b/trl/trainer/grpo_trainer.py#L965C39-L965C66). This approach quickly becomes a major bottleneck — generation is slow and inefficient, particularly for large batches or models. As a result, training times increase significantly, and overall efficiency drops. To address this, we turn to vLLM, which enables much faster and more scalable generation, helping eliminate this bottleneck in online methods.

## 🤔 How does vLLM solve the slow generation issue?

If you've ever done autoregressive decoder training, you know all the input tokens to the LLM produce their attention key and value tensors, and these tensors are kept in GPU memory to later generate subsequent tokens based on them. These cached key and value tensors are often referred to as the KV cache. However, storing the KV cache occupies a lot of memory, so vLLM uses a technique called **PagedAttention** to solve this problem. PagedAttention, which is inspired by the OS’s virtual memory concept, stores continuous keys and values in **non-contiguous memory space**, which is much more efficient. The details of this are beyond the scope of this document, but in short, it allows the model to store the keys and values in a more efficient way, reducing the memory footprint and speeding up the generation process. If you are interested, make sure to check out the [vLLM PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) for more details.

## 🤔 What exactly happens when you run `trl vllm-serve --model <model_name>`?

When you run for example

```sh
trl vllm-serve --model Qwen/Qwen2.5-7B --tensor-parallel-size 1 --data-parallel-size 4
```

the following happens:

![vllm](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/vllm-doc.png)

1. vLLM first spawns multiple workers to handle incoming requests in parallel. The number of workers is determined by multiplying the `--tensor-parallel-size` and `--data-parallel-size` values. In this example, it spawns 4 workers (1 × 4).
Each worker operates independently and processes a chunk of the incoming requests — which are basically the prompts sent to the server for generation. A key point to understand is that these 4 workers are running in parallel, and each one is responsible for handling a subset of the total incoming load.

2. Once the incoming requests (prompts) are distributed across the workers, the model starts generating completions. Internally, the model’s weights are split across multiple GPUs based on the `--tensor-parallel-size` argument — this is how tensor parallelism is handled. Meanwhile, data parallelism (controlled by `--data-parallel-size`) ensures that different sets of requests are processed independently across the workers. In short: tensor parallelism splits the model across GPUs, and data parallelism splits the batch of requests across different model replicas.

3. Although the GPUs process requests independently and in parallel, they still need to communicate with each other. Remember that each GPU handles only a slice of the incoming prompts (for example, with 4 GPUs and 8 prompts using `--data-parallel-size=4`, each GPU processes 2 prompts).
This GPU-to-GPU communication is managed efficiently by NVIDIA’s NCCL library. The communication mainly ensures that each GPU gets its correct portion of the incoming requests — it’s lightweight and doesn’t interfere with generation itself.
Separately, the number of completions to generate per prompt is controlled by the `num_generations` setting in the GRPO config. For instance, if you set `num_generations=2` (like in the picture above), each prompt will have 2 completions. So, with 8 prompts and `num_generations=2`, you would end up with 16 completions total — regardless of the number of GPUs or parallelism settings.

## 🥸 More detail on what happens under the hood when running the server

* The vLLM server starts by running the command: `trl vllm-serve --model Qwen/Qwen2.5-7B`.
* Once the server is running, it generates completions based on requests from the client (trainer) using `vllm_client.generate` [here](https://github.com/huggingface/trl/blob/cc044e35b285be7dc062764b3364e1e684db4c7c/trl/trainer/grpo_trainer.py#L1025-L1035).
* The client (trainer) then requests these completions from the server.
* These completions are used to compute the reward signal.
* Based on the reward signal and the model’s output, the loss is computed, and the backward pass is performed to update the model’s weights.
* **Note**: The server only handles completion generation — it doesn’t train the model. Therefore, the model’s weights aren’t updated on the server. Once the backward pass is complete, the client sends the updated weights to the server using `vllm_client.update_named_param(name, param.data)`.

When using vLLM, ensure the GPUs assigned for training and generation are separate to avoid resource conflicts. For instance, if you plan to use 4 GPUs for training and another 4 for vLLM generation, you can specify GPU allocation for training using `CUDA_VISIBLE_DEVICES`. See the example below:

* **Set GPUs *0–3* for vLLM generation:** Assume `CUDA_VISIBLE_DEVICES=0,1,2,3` are allocated for vLLM generation.

```sh
trl vllm-serve --model <model_name> --tensor-parallel-size 1 --data-parallel-size 4
```

* **And GPUs *4–7* for training:** If you do not set the `CUDA_VISIBLE_DEVICES` environment variable, the training script will use all available GPUs by default, which may lead to resource conflicts. To avoid this, you can specify which GPUs to use for training. For example, if you want to use GPUs 4–7 for training, set the environment variable as follows:

```sh
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch train.py
```

## 🍷 More customization options with vLLM?

You can customize the server configuration by passing additional arguments.

```
$ trl vllm-serve --help
usage: trl vllm-serve [-h] --model MODEL [--revision REVISION] [--tensor_parallel_size TENSOR_PARALLEL_SIZE]
                      [--data_parallel_size DATA_PARALLEL_SIZE] [--host HOST] [--port PORT]
                      [--gpu_memory_utilization GPU_MEMORY_UTILIZATION] [--dtype DTYPE] [--max_model_len MAX_MODEL_LEN]
                      [--enable_prefix_caching ENABLE_PREFIX_CACHING] [--enforce_eager ENFORCE_EAGER] [--log_level LOG_LEVEL]

options:
  -h, --help            Show this help message and exit
  --model MODEL         Model name or path to load the model from. (default: None)
  --revision REVISION   Revision to use for the model. If not specified, the default branch will be used. (default: None)
  --tensor_parallel_size TENSOR_PARALLEL_SIZE, --tensor-parallel-size TENSOR_PARALLEL_SIZE
                        Number of tensor parallel workers to use. (default: 1)
  --data_parallel_size DATA_PARALLEL_SIZE, --data-parallel-size DATA_PARALLEL_SIZE
                        Number of data parallel workers to use. (default: 1)
  --host HOST           Host address to run the server on. (default: 0.0.0.0)
  --port PORT           Port to run the server on. (default: 8000)
  --gpu_memory_utilization GPU_MEMORY_UTILIZATION, --gpu-memory-utilization GPU_MEMORY_UTILIZATION
                        Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache on the device
                        dedicated to generation powered by vLLM. Higher values will increase the KV cache size and thus improve the
                        model's throughput. However, if the value is too high, it may cause out-of-memory (OOM) errors during
                        initialization. (default: 0.9)
  --dtype DTYPE         Data type to use for vLLM generation. If set to 'auto', the data type will be automatically determined based on
                        the model configuration. Find the supported values in the vLLM documentation. (default: auto)
  --max_model_len MAX_MODEL_LEN, --max-model-len MAX_MODEL_LEN
                        If set, the `max_model_len` to use for vLLM. This can be useful when running with reduced
                        `vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model context
                        size, which might be much larger than the KV cache, leading to inefficiencies. (default: None)
  --enable_prefix_caching ENABLE_PREFIX_CACHING, --enable-prefix-caching ENABLE_PREFIX_CACHING
                        Whether to enable prefix caching in vLLM. If set to `True`, ensure that the model and the hardware support this
                        feature. (default: None)
  --enforce_eager ENFORCE_EAGER, --enforce-eager ENFORCE_EAGER
                        Whether to enforce eager execution. If set to `True`, we will disable CUDA graph and always execute the model
                        in eager mode. If `False` (default behavior), we will use CUDA graph and eager execution in hybrid. (default:
                        None)
  --log_level LOG_LEVEL, --log-level LOG_LEVEL
                        Log level for uvicorn. Possible choices: 'critical', 'error', 'warning', 'info', 'debug', 'trace'. (default:
                        info)
```

## 🥳 Okay, now that we have the server running, how can we use it to generate completions?

Run the training script and pass `use_vllm=True` in the training arguments:

```python
from trl import GRPOConfig

training_args = GRPOConfig(..., use_vllm=True)
```

## 💆🏻‍♀️ What's the best distributed setup?

![](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/tp_dp_throughput_8_gpus.png)
![](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/tp_dp_throughput_4_gpus.png)

First and foremost, always remember that the optimal setup depends on:

* The model size
* The number of GPUs you have
* The GPU memory size
* The batch size you are using
* The number of requests you are sending to the server (prompts)
* The `max_model_len` you are using (this is the max length of the input sequence that the model can process, a.k.a. the context window size)
* The number of completions you are generating for each request (`num_generations`)

Given these factors, our experiments on the Qwen model family (3B, 7B, 14B, 32B) using 8 H100 GPUs show that:

* For reasonable-sized models (3B–14B) and a moderate context window (`max_len < 8k`), using full capacity for data parallelism gives better throughput. The setup `(tp=1, dp=8)` yields the best results.
* For larger models (32B) and longer context windows (`max_len > 8k`), a smaller DP size combined with some model-side parallelism performs better. For example, `(tp=2, dp=4)` is a good setup for 32B models with a larger context window.
