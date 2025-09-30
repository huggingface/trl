# LoRA Without Regret

Recent research from the team at [Thinking Machines Lab](https://thinkingmachines.ai/blog/lora/) (Schulman et al., 2025) shows that **LoRA can match full fine-tuning performance** when configured correctly, while using only ~67% of the compute. These findings are exciting to TRL users because they're straight forward to implement and can improve model performance on smaller budgets.

This guide provides simple instructions to reproduce the results of the blog post in TRL.

## Benefits of LoRA over full fine-tuning

First of all, let's remind ourselves of the benefits of [LoRA over full fine-tuning](https://huggingface.co/docs/trl/en/peft_integration).

LoRA trains an adapter layer on top of the base model, which contains significantly fewer parameters than the base model itself. This allows us to train the model on less GPU memory. It has generally been accepted that this comes with a trade-off in performance. The [blog post](https://thinkingmachines.ai/blog/lora/) proposes that with the correct configuration, LoRA can overcome this tradeoff and match full fine-tuning performance.

## Key findings in optimizing LoRA

Let's dive into the key findings of the blog post one by one and see how we can implement them in TRL scripts. Below, we will reproduce the results of the blog post using complete the TRL scripts that you can run locally or on Hugging Face Jobs.

### 1. *LoRA performs better when applied to all weight matrices*

The authors recommend applying LoRA to all weight matrices instead of attention-only LoRA targeted at attention layers, and this is not overcome by increasing the rank. In TRL script, we could use `--lora_target_modules all-linear` to apply LoRA to all weight matrices.

![all layers](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lora_without_regret/1.png)

Attention-only LoRA underperforms even when using higher rank to match parameter count.

### 2. *We can estimate trainable parameters from dataset size to determine LoRA rank*

The blog post recommends choosing LoRA rank based on task and dataset size. LoRA rank controls the number of trainable parameters in the LoRA adapter. And the post proposes that LoRA works well when the number of parameters exceeds the amount of information to be learned, which we should estimate from the dataset size.

![learning rate](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lora_without_regret/3.png)

In TRL script, we could use `--lora_r` to set the rank and adapt it based on the task and dataset we're training on. The blog post recommends the following ranks based on the task and dataset size:

| Task Type | Dataset Size | Recommended Rank |
|-----------|-------------|------------------|
| **SFT** - Small instruction | <10K examples | 32-64 |
| **SFT** - Medium instruction | 10K-1M examples | 64-128 |
| **SFT** - Large reasoning | >1M examples | 256+ |
| **RL** - All tasks | Any size | 8-32 |

Reinforcement learning requires minimal capacity, so we can use lower ranks. This is because policy gradient algorithms learn only ~1 bit per episode, requiring minimal capacity.

### 3. *"FullFT and high-rank LoRAs have similar learning curves"*

Counter-intuitively, the blog post recommends using similar learning rates to full fine-tuning. In TRL script, we could use `--learning_rate` to set the learning rate. The 1/r scaling in LoRA makes optimal learning rate approximately rank-independent.

![learning rate](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lora_without_regret/2.png)

### 4. *"In some scenarios, LoRA is less tolerant of large batch sizes than full fine-tuning."*

The blog post recommends using effective batch size < 256 because the authors found LoRA to be less tolerant of large batch sizes. This could not be mitigated by increasing the LoRA rank. In TRL script, we could use `--per_device_train_batch_size` and `--gradient_accumulation_steps` to set the batch size.

![learning rate](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lora_without_regret/4.png)

## Examples with TRL

Those are the core findings of the blog post. Let's implement them in TRL scripts to train LoRA adapters.

### Supervised Fine-Tuning (SFT)

The blog post performs SFT on a range of models and datasets from the Hub, which we can reproduce in TRL.

| Model | Dataset |
|-------|---------|
| [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B) | [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) |
| [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B) | [open-thoughts/OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) |
| [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B) | [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) |
| [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B) | [open-thoughts/OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) |

<hfoptions id="sft">
<hfoption id="jobs">

```bash

# Medium dataset (Tulu3) - use rank 128
# TODO: add hf jobs command
```

To use Hugging Face Jobs, you will need to be logged in to the Hugging Face Hub (`hf auth login`) and have a [Pro](https://hf.co/pro), [Team](https://hf.co/enterprise), or [Enterprise](https://hf.co/enterprise) plan. Check out the [Jobs documentation](https://huggingface.co/docs/huggingface_hub/en/guides/jobs) for more details.

</hfoption>
<hfoption id="local">

```bash

# Medium dataset (Tulu3) - use rank 128
# TODO: local command
```

To run th script locally, you will need to have `uv` installed. Check out the [uv documentation](https://docs.astral.sh/uv/) for more details.

</hfoption>
</hfoptions>

Once training starts, you can monitor the progress in [Trackio](https://huggingface.co/trackio) which will log the url.

<!-- TODO: @burtenshaw - add trackio iframe -->

### Reinforcement Learning (GRPO)

The blog post performs GRPO on a range of models and datasets from the Hub, and once again we can reproduce the results in TRL.

<!-- TODO: @edbeeching - describe rl function -->


| Model | Dataset |
|-------|---------|
| [Llama-3.1-8B-Base](https://huggingface.co/meta-llama/Llama-3.2-1B) | [GSM8k](https://huggingface.co/datasets/openai/gsm8k) |
| [Llama-3.1-8B-Base](https://huggingface.co/meta-llama/Llama-3.2-1B) | [DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K) |
| [Qwen3-8b-base](https://huggingface.co/Qwen/Qwen3-8b-base) | [DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K) |

<hfoptions id="sft">
<hfoption id="jobs">

```bash

# Medium dataset (Tulu3) - use rank 128
# TODO: add hf jobs command
```

To use Hugging Face Jobs, you will need to be logged in to the Hugging Face Hub (`hf auth login`) and have a [Pro](https://hf.co/pro), [Team](https://hf.co/enterprise), or [Enterprise](https://hf.co/enterprise) plan. Check out the [Jobs documentation](https://huggingface.co/docs/huggingface_hub/en/guides/jobs) for more details.

</hfoption>
<hfoption id="local">

```bash

# Medium dataset (Tulu3) - use rank 128
# TODO: local command
```

To run th script locally, you will need to have `uv` installed. Check out the [uv documentation](https://docs.astral.sh/uv/) for more details.

</hfoption>
</hfoptions>

<!-- TODO: @burtenshaw - add trackio iframe -->

## Scripts

The above commands are both implement as custom scripts in TRL based on the configurations recommended by the blog post.

- [`sft_lora.py`]() - Supervised fine-tuning with LoRA best practices
- [`grpo_lora.py`]() - Reinforcement learning with LoRA best practices

<!-- TODO: @burtenshaw - add scripts links -->

## Citation

```bibtex
@article{schulman2025lora,
  author = {John Schulman and Thinking Machines Lab},
  title = {LoRA Without Regret},
  journal = {Thinking Machines Lab: Connectionism},
  year = {2025},
  note = {https://thinkingmachines.ai/blog/lora/},
  doi = {10.64434/tml.20250929},
}
```
