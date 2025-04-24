# 🚀 Speeding Up Training TRL x vLLM 

# 🎬 Flashback: why do we need to use vLLM in online methods?

Online methods such as GRPO or Online DPO require the model to generate completions. Because these completions are used to compute the reward signal, they need to be generated at regular intervals during training. This is typically done every `gradient_accumulation_steps * num_iterations` steps, where `num_iterations` is the number of iterations between two gradient updates. Now, the problem is that generating completions is a time-consuming process, especially when using large models. The reason of the time-consuming nature of the generation process is by default, this is done using the [(unwrapped)model's `generate` method](https://github.com/huggingface/trl/blob/f3e8c2304428ef16e9ae5de9e5741ed84d533b7b/trl/trainer/grpo_trainer.py#L965C39-L965C66), (ofcourse if `use_vllm` is set to False). This `unwrapped_model.generate` method is thechnically a synchronous function, meaning that it will block the execution of the program until the generation is complete. This can lead to inefficiencies, especially when generating large batches of completions or when using large models. Therefore, the generation process can become a bottleneck in the training process, leading to longer training times and reduced efficiency. So this is why we need to think of a better way to do this which is using vLLM for faster generation. 

# How does vLLM solve the slow generation issue?
if you've ever done autoregressive decoder training, you know  all the input tokens to the LLM produce their attention key and value tensors, and these tensors are kept in GPU memory to later generate next tokens (Q) based on them. These cached key and value tensors are often referred to as KV cache.  However, this storing is really a pain as it occupies a lot of memory. 
So here is the secret sauce of vLLM, it uses a technique called PagedAttention to solve this problem. PagedAttention , which is inspired by the OS’s virtual memory concept stores continuous keys and values in **non-contiguous memory space** which is way more efficient. The detail of this is beyond the scope of this document, but in short, it allows the model to store the keys and values in a more efficient way, reducing the memory footprint and speeding up the generation process. If you are interested, make sure to check out the [vLLM PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) for more details.


# How to use vLLM in practice for generation in online methods in TRL?

1. To use [vLLM](https://github.com/vllm-project/vllm), first install it using:

```bash
pip install vllm
```

or 

```bash
pip install "trl[vllm]"
```

<hfoptions id="vllm examples">
<hfoption id="Online DPO">

</hfoption>
<hfoption id="GRPO">

2. Then, **start a vLLM server** by running:

```bash
trl vllm-serve --model <model_name>
```

Then, run the training script and pass `use_vllm=True` in the training arguments.

```python
from trl import GRPOConfig

training_args = GRPOConfig(..., use_vllm=True)
```

You can customize the server configuration by passing additional arguments. For more information, see [vLLM integration](vllm_integration).

# Okay, now that we have the server running, how can we use it to generate completions? 

Then, run the training script and pass `use_vllm=True` in the training arguments.

```python
from trl import GRPOConfig

training_args = GRPOConfig(..., use_vllm=True)
```

# TL;DR 
First run the server by; (this example allocate 4 GPUs for vLLM generation)
```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 trl vllm-serve --model Qwen/Qwen2.5-7B
```  
Then, run the training script by passing `use_vllm=True` in the training arguments (this example allocate 4 GPUs for training) by;
  
```sh
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch train.py
```  

</Tip>

</hfoption>
</hfoptions>
