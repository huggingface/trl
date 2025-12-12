# LoRA Without Regret

Recent research from the team at [Thinking Machines Lab](https://thinkingmachines.ai/blog/lora/) (Schulman et al., 2025) shows that **LoRA can match full fine-tuning performance** when configured correctly, while using only ~67% of the compute. These findings are exciting to TRL users because they're straightforward to implement and can improve model performance on smaller budgets.

This guide provides simple instructions to reproduce the results of the blog post in TRL.

> [!TIP]
> It is recommended to read the blog post before following this guide, or to consult both resources in parallel for best results.

## Benefits of LoRA over full fine-tuning

First of all, let's remind ourselves of the benefits of [LoRA over full fine-tuning](https://huggingface.co/docs/trl/en/peft_integration).

LoRA adds adapter layers on top of the base model, which contains significantly fewer parameters than the base model itself. This design reduces GPU memory requirements and enables more efficient training. As described in the [blog](https://thinkingmachines.ai/blog/lora/), this approach was originally thought to involve a performance trade-off, although careful configuration can overcome this trade-off and match full fine-tuning performance.  

## Examples with TRL

Let's implement and train LoRA adapters in TRL scripts based on the core findings of the blog post. Afterwards, we'll revisit each finding in light of the TRL results.

### Supervised Fine-Tuning (SFT)

The blog post performs SFT on a range of models and datasets from the Hub, which we can reproduce in TRL.

| Model | Dataset |
| --- | --- |
| [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B) | [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) |
| [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B) | [open-thoughts/OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) |
| [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B) | [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) |
| [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B) | [open-thoughts/OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) |

<hfoptions id="sft">
<hfoption id="python">

We can integrate these findings with the TRL Python API like so:

```python

from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

dataset = load_dataset("open-thoughts/OpenThoughts-114k", split="train")

peft_config = LoraConfig(r=256, lora_alpha=16, target_modules="all-linear")

training_args = SFTConfig(
    learning_rate=2e-4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    report_to=["trackio"],
)

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-3B-Instruct",
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_args,
)

trainer.train()

```

</hfoption>
<hfoption id="jobs">

```bash

hf jobs uv run \
    --flavor a100-large \
    --timeout 8h \
    --secrets HF_TOKEN \
    "https://raw.githubusercontent.com/huggingface/trl/main/trl/scripts/sft.py" \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --dataset_name open-thoughts/OpenThoughts-114k \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --use_peft \
    --lora_r 256 \
    --lora_alpha 16 \
    --lora_target_modules all-linear \
    --output_dir Qwen2.5-3B-OpenThoughts-LoRA \
    --report_to trackio \
    --push_to_hub

```

To use Hugging Face Jobs, you will need to be logged in to the Hugging Face Hub (`hf auth login`) and have a [Pro](https://hf.co/pro), [Team](https://hf.co/enterprise), or [Enterprise](https://hf.co/enterprise) plan. Check out the [Jobs documentation](https://huggingface.co/docs/huggingface_hub/en/guides/jobs) for more details.

</hfoption>
<hfoption id="local">

```bash

uv run "https://raw.githubusercontent.com/huggingface/trl/main/trl/scripts/sft.py" \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --dataset_name open-thoughts/OpenThoughts-114k \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing \
    --eval_strategy no \
    --use_peft \
    --lora_r 256 \
    --lora_alpha 16 \
    --lora_target_modules all-linear \
    --output_dir Qwen2.5-3B-OpenThoughts-LoRA \
    --report_to trackio \
    --push_to_hub

```

To run the script locally, you will need to have `uv` installed. Check out the [uv documentation](https://docs.astral.sh/uv/) for more details.

</hfoption>
</hfoptions>

Once training starts, you can monitor the progress in [Trackio](https://huggingface.co/trackio), which will log the URL.

### Reinforcement Learning (GRPO)

The blog post performs GRPO on a range of models and datasets from the Hub, and once again we can reproduce the results in TRL.

| Model | Dataset |
| --- | --- |
| [Llama-3.1-8B-Base](https://huggingface.co/meta-llama/Llama-3.2-1B) | [GSM8k](https://huggingface.co/datasets/openai/gsm8k) |
| [Llama-3.1-8B-Base](https://huggingface.co/meta-llama/Llama-3.2-1B) | [DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K) |
| [Qwen3-8b-base](https://huggingface.co/Qwen/Qwen3-8b-base) | [DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K) |

For reinforcement learning, the blog uses a math reasoning task that we can reproduce as a Python function.

<hfoptions id="grpo">
<hfoption id="python">

We can implement these recommendations with the TRL Python API like so:

```python

from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import reasoning_accuracy_reward

dataset = load_dataset("HuggingFaceH4/OpenR1-Math-220k-default-verified", split="train")

peft_config = LoraConfig(
    r=1,
    lora_alpha=32,
    target_modules="all-linear"
)

training_args = GRPOConfig(
    learning_rate=5e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    num_generations=8,
    generation_batch_size=8,
    report_to=["trackio"],
)

trainer = GRPOTrainer(
    model="Qwen/Qwen3-0.6B",
    reward_funcs=reasoning_accuracy_reward,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
)

trainer.train()

```

> [!WARNING]
> This snippet skips the reward function which is defined above to keep the example concise.

</hfoption>
<hfoption id="jobs">

```bash

hf jobs uv run \
    --flavor a100-large \
    --timeout 4h \
    --secrets HF_TOKEN \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "https://huggingface.co/datasets/burtenshaw/lora-without-regrets/resolve/main/grpo.py" \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --dataset_name HuggingFaceH4/OpenR1-Math-220k-default-verified \
    --output_dir grpo-full-qwen3-0.6b \
    --learning_rate 1.0e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.0 \
    --max_grad_norm 1.0 \
    --beta 0.0 \
    --max_completion_length 4096 \
    --num_generations 16 \
    --generation_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --per_device_train_batch_size 1 \
    --num_train_epochs 1 \
    --lora_r 1 \
    --lora_alpha 32 \
    --lora_dropout 0.0 \
    --lora_target_modules all-linear \
    --vllm_mode colocate \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 1 \
    --logging_steps 1 \
    --max_steps 200 \
    --report_to trackio
```

To use Hugging Face Jobs, you will need to be logged in to the Hugging Face Hub (`hf auth login`) and have a [Pro](https://hf.co/pro), [Team](https://hf.co/enterprise), or [Enterprise](https://hf.co/enterprise) plan. Check out the [Jobs documentation](https://huggingface.co/docs/huggingface_hub/en/guides/jobs) for more details.

</hfoption>
<hfoption id="local">

```bash
uv run "https://huggingface.co/datasets/burtenshaw/lora-without-regrets/resolve/main/grpo.py" \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --dataset_name HuggingFaceH4/OpenR1-Math-220k-default-verified \
    --output_dir grpo-full-qwen3-0.6b \
    --learning_rate 1.0e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.0 \
    --max_grad_norm 1.0 \
    --beta 0.0 \
    --max_completion_length 4096 \
    --num_generations 16 \
    --generation_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --per_device_train_batch_size 1 \
    --num_train_epochs 1 \
    --lora_r 1 \
    --lora_alpha 32 \
    --lora_dropout 0.0 \
    --lora_target_modules all-linear \
    --vllm_mode colocate \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 1 \
    --logging_steps 1 \
    --max_steps 200 \
    --report_to trackio
```

To run the script locally, you will need to have `uv` installed. Check out the [uv documentation](https://docs.astral.sh/uv/) for more details.

</hfoption>
</hfoptions>

The reinforcement learning script with GRPO is implemented as a custom script in TRL, which uses the reward function shown above. You can review it at [`grpo.py`](https://huggingface.co/datasets/burtenshaw/lora-without-regrets/blob/main/grpo.py) - Reinforcement learning with LoRA best practices

## Key findings in optimizing LoRA

The authors recommend applying LoRA to all weight matrices rather than limiting it to attention layers, as increasing the rank does not compensate for this restriction. In TRL, this can be configured using `--lora_target_modules all-linear` to apply LoRA to all weight matrices.

We were able to reproduce the results of the blog post using TRL and the SmolLM3 model. We trained the model for 500 steps on the [Math 220k dataset](https://huggingface.co/datasets/HuggingFaceH4/OpenR1-Math-220k-default-verified) with the reward function and configuration above. As you can see in the figure below, the LoRA model's average train reward curve matches the full fine-tuning curve.

![train reward](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lora_without_regret/5.png)

And most importantly, the LoRA model uses significantly less memory than the full fine-tuning model, as we can see in the figure below.

![memory usage](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lora_without_regret/6.png)

Here are the parameters we used to train the above models

| Parameter | LoRA | Full FT |
| --- | --- | --- |
| `--model_name_or_path` | HuggingFaceTB/SmolLM3-3B | HuggingFaceTB/SmolLM3-3B |
| `--dataset_name` | HuggingFaceH4/OpenR1-Math-220k-default-verified | HuggingFaceH4/OpenR1-Math-220k-default-verified |
| `--learning_rate` | 1.0e-5 | 1.0e-6 |
| `--max_prompt_length` | 1024 | 1024 |
| `--max_completion_length` | 4096 | 4096 |
| `--lora_r` | 1 | - |
| `--lora_alpha` | 32 | - |
| `--lora_dropout` | 0.0 | - |
| `--lora_target_modules` | all-linear | - |

Let's break down the key findings of the blog post and how we were able to reproduce them.

### 1. *LoRA performs better when applied to all weight matrices*

The authors recommend applying LoRA to all weight matrices rather than limiting it to attention layers, as increasing the rank does not compensate for this restriction.

![all layers](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lora_without_regret/1.png)

Attention-only LoRA underperforms even when using a higher rank to match parameter count. In TRL, this can be configured using `--lora_target_modules all-linear` to apply LoRA to all weight matrices.  In Python, we can do this like so:

```python
from peft import LoraConfig  

peft_config = LoraConfig(target_modules="all-linear")  
```

### 2. *The adapter needs sufficient capacity to learn from the dataset*

The blog post recommends using a sufficient LoRA rank to learn from the dataset. The rank determines the number of trainable parameters in the LoRA adapter. Therefore, "For datasets that exceed LoRA capacity, LoRA underperforms FullFT".

![learning rate](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lora_without_regret/3.png)

In the TRL script, we could use `--lora_r` to set the rank and adapt it based on the task and dataset we're training on. The blog post recommends the following ranks based on the task and dataset size:

Reinforcement learning tasks typically require lower capacity, so smaller LoRA ranks can be used. This is because policy gradient algorithms extract roughly ~1 bit of information per episode, demanding minimal parameter capacity.  

The blog post defines the ideal dataset size for LoRA to match full fine-tuning as "Post-training scale". Which we can use to determine the recommended rank for SFT and RL LoRAs as:

| Task Type | Dataset Size | Recommended Rank |
| --- | --- | --- |
| **SFT** | Post-training scale | 256 |
| **RL** | Any size | 1-32 |

### 3. *"FullFT and high-rank LoRAs have similar learning curves"*

Counterintuitively, the blog post recommends using a higher learning rate than for full fine-tuning. In the table above, we used 1.0e-5 for LoRA and 1.0e-6 for full fine-tuning. In the TRL script, we could use `--learning_rate` to set the learning rate. The  \\( \frac{1}{r} \\) scaling in LoRA makes the optimal learning rate approximately rank-independent.

![learning rate](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lora_without_regret/2.png)

### 4. *"In some scenarios, LoRA is less tolerant of large batch sizes than full fine-tuning."*

The blog post recommends using an effective batch size < 32 because the authors found LoRA to be less tolerant of large batch sizes. This could not be mitigated by increasing the LoRA rank. In the TRL script, we could use `--per_device_train_batch_size` and `--gradient_accumulation_steps` to set the batch size.

![learning rate](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lora_without_regret/4.png)

## Takeaways

Using TRL, you can efficiently implement LoRA adapters to match full fine-tuning performance, applying the core insights (targeting all weight matrices, choosing the right rank, and managing batch size and learning rate) without the heavy compute cost of FullFT.

## Citation

```bibtex
@article{schulman2025lora,  
    title        = {{LoRA Without Regret}},  
    author       = {John Schulman and Thinking Machines Lab},  
    year         = 2025,  
    journal      = {Thinking Machines Lab: Connectionism},  
    doi          = {10.64434/tml.20250929},  
    note         = {https://thinkingmachines.ai/blog/lora/}  
}  
```
