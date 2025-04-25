# vLLM Integration

<Tip warning={true}>

Section under construction. Feel free to contribute!

</Tip>

## TRL vLLM server

TRL provides a way to speedup generation using a dedicated vLLM server. 

```sh
$ trl vllm-serve --help
usage: trl vllm-serve [-h] --model MODEL [--revision REVISION] [--tensor_parallel_size TENSOR_PARALLEL_SIZE] [--host HOST]
                      [--port PORT] [--gpu_memory_utilization GPU_MEMORY_UTILIZATION] [--dtype DTYPE]
                      [--max_model_len MAX_MODEL_LEN] [--enable_prefix_caching ENABLE_PREFIX_CACHING]

options:
  -h, --help            Show this help message and exit
  --model MODEL         Model name or path to load the model from. (default: None)
  --revision REVISION   Revision to use for the model. If not specified, the default branch will be used. (default: None)
  --tensor_parallel_size TENSOR_PARALLEL_SIZE, --tensor-parallel-size TENSOR_PARALLEL_SIZE
                        Number of tensor parallel workers to use. (default: 1)
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
```

### Find the best distributed setup

![](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/tp_dp_throughput_8_gpus.png)

![](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/tp_dp_throughput_4_gpus.png)