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

</hfoption>
</hfoptions>
