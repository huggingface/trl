# Command Line Interfaces (CLIs)

TRL provides a powerful command-line interface (CLI) to fine-tune large language models (LLMs) using methods like Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and more. The CLI abstracts away much of the boilerplate, letting you launch training jobs quickly and reproducibly.

## Commands

Currently supported commands are:

### Training Commands

- `trl dpo`: fine-tune a LLM with DPO
- `trl grpo`: fine-tune a LLM with GRPO
- `trl kto`: fine-tune a LLM with KTO
- `trl reward`: train a Reward Model
- `trl rloo`: fine-tune a LLM with RLOO
- `trl sft`: fine-tune a LLM with SFT

### Other Commands

- `trl env`: get the system information
- `trl vllm-serve`: serve a model with vLLM

## Fine-Tuning with the TRL CLI

### Basic Usage

You can launch training directly from the CLI by specifying required arguments like the model and dataset:

<hfoptions id="trainer">
<hfoption id="SFT">

```bash
trl sft \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name stanfordnlp/imdb
```

</hfoption>
<hfoption id="DPO">

```bash
trl dpo \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name anthropic/hh-rlhf
```

</hfoption>
<hfoption id="Reward">

```bash
trl reward \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name trl-lib/ultrafeedback_binarized
```

</hfoption>
<hfoption id="GRPO">

```bash
trl grpo \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name HuggingFaceH4/Polaris-Dataset-53K \
  --reward_funcs accuracy_reward
```

</hfoption>
<hfoption id="RLOO">

```bash
trl rloo \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name HuggingFaceH4/Polaris-Dataset-53K \
  --reward_funcs accuracy_reward
```

</hfoption>
<hfoption id="KTO">

```bash
trl kto \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name trl-lib/kto-mix-14k
```

</hfoption>
</hfoptions>

### Using Configuration Files

To keep your CLI commands clean and reproducible, you can define all training arguments in a YAML configuration file:

<hfoptions id="trainer">
<hfoption id="SFT">

```yaml
# sft_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
dataset_name: stanfordnlp/imdb
```

Launch with:

```bash
trl sft --config sft_config.yaml
```

</hfoption>
<hfoption id="DPO">

```yaml
# dpo_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
dataset_name: anthropic/hh-rlhf
```

Launch with:

```bash
trl dpo --config dpo_config.yaml
```

</hfoption>
<hfoption id="Reward">

```yaml
# reward_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
dataset_name: trl-lib/ultrafeedback_binarized
```

Launch with:

```bash
trl reward --config reward_config.yaml
```

</hfoption>
<hfoption id="GRPO">

```yaml
# grpo_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
dataset_name: HuggingFaceH4/Polaris-Dataset-53K
reward_funcs:
  - accuracy_reward
```

Launch with:

```bash
trl grpo --config grpo_config.yaml
```

</hfoption>
<hfoption id="RLOO">

```yaml
# rloo_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
dataset_name: HuggingFaceH4/Polaris-Dataset-53K
reward_funcs:
  - accuracy_reward
```

Launch with:

```bash
trl rloo --config rloo_config.yaml
```

</hfoption>
<hfoption id="KTO">

```yaml
# kto_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
dataset_name: trl-lib/kto-mix-14k
```

Launch with:

```bash
trl kto --config kto_config.yaml
```

</hfoption>
</hfoptions>

### Scaling Up with Accelerate

TRL CLI natively supports [🤗 Accelerate](https://huggingface.co/docs/accelerate), making it easy to scale training across multiple GPUs, machines, or use advanced setups like DeepSpeed — all from the same CLI.

You can pass any `accelerate launch` arguments directly to `trl`, such as `--num_processes`. For more information see [Using accelerate launch](https://huggingface.co/docs/accelerate/en/basic_tutorials/launch#using-accelerate-launch).

<hfoptions id="trainer">
<hfoption id="SFT">

```bash
trl sft \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name stanfordnlp/imdb \
  --num_processes 4
```

or, with a config file:

```yaml
# sft_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
dataset_name: stanfordnlp/imdb
num_processes: 4
```

Launch with:

```bash
trl sft --config sft_config.yaml
```

</hfoption>
<hfoption id="DPO">

```bash
trl dpo \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name anthropic/hh-rlhf \
  --num_processes 4
```

or, with a config file:

```yaml
# dpo_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
dataset_name: anthropic/hh-rlhf
num_processes: 4
```

Launch with:

```bash
trl dpo --config dpo_config.yaml
```

</hfoption>
<hfoption id="Reward">

```bash
trl reward \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name trl-lib/ultrafeedback_binarized \
  --num_processes 4
```

or, with a config file:

```yaml
# reward_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
dataset_name: trl-lib/ultrafeedback_binarized
num_processes: 4
```

Launch with:

```bash
trl reward --config reward_config.yaml
```

</hfoption>
<hfoption id="GRPO">

```bash
trl grpo \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name HuggingFaceH4/Polaris-Dataset-53K \
  --reward_funcs accuracy_reward \
  --num_processes 4
```

or, with a config file:

```yaml
# grpo_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
dataset_name: HuggingFaceH4/Polaris-Dataset-53K
reward_funcs:
  - accuracy_reward
num_processes: 4
```

Launch with:

```bash
trl grpo --config grpo_config.yaml
```

</hfoption>
<hfoption id="RLOO">

```bash
trl rloo \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name HuggingFaceH4/Polaris-Dataset-53K \
  --reward_funcs accuracy_reward \
  --num_processes 4
```

or, with a config file:

```yaml
# rloo_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
dataset_name: HuggingFaceH4/Polaris-Dataset-53K
reward_funcs:
  - accuracy_reward
num_processes: 4
```

Launch with:

```bash
trl rloo --config rloo_config.yaml
```

</hfoption>
<hfoption id="KTO">

```bash
trl kto \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name trl-lib/kto-mix-14k \
  --num_processes 4
```

or, with a config file:

```yaml
# kto_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
dataset_name: trl-lib/kto-mix-14k
num_processes: 4
```

Launch with:

```bash
trl kto --config kto_config.yaml
```

</hfoption>
</hfoptions>

### Using `--accelerate_config` for Accelerate Configuration

The `--accelerate_config` flag lets you easily configure distributed training with [🤗 Accelerate](https://github.com/huggingface/accelerate). This flag accepts either:

- the name of a predefined config profile (built into TRL), or
- a path to a custom Accelerate YAML config file.

#### Predefined Config Profiles

TRL provides several ready-to-use Accelerate configs to simplify common training setups:

| Name | Description |
| --- | --- |
| `fsdp1` | Fully Sharded Data Parallel Stage 1 |
| `fsdp2` | Fully Sharded Data Parallel Stage 2 |
| `zero1` | DeepSpeed ZeRO Stage 1 |
| `zero2` | DeepSpeed ZeRO Stage 2 |
| `zero3` | DeepSpeed ZeRO Stage 3 |
| `multi_gpu` | Multi-GPU training |
| `single_gpu` | Single-GPU training |

To use one of these, just pass the name to `--accelerate_config`. TRL will automatically load the corresponding config file from `trl/accelerate_config/`.

#### Example Usage

<hfoptions id="trainer">
<hfoption id="SFT">

```bash
trl sft \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name stanfordnlp/imdb \
  --accelerate_config zero2  # or path/to/my/accelerate/config.yaml
```

or, with a config file:

```yaml
# sft_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
dataset_name: stanfordnlp/imdb
accelerate_config: zero2  # or path/to/my/accelerate/config.yaml
```

Launch with:

```bash
trl sft --config sft_config.yaml
```

</hfoption>
<hfoption id="DPO">

```bash
trl dpo \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name anthropic/hh-rlhf \
  --accelerate_config zero2  # or path/to/my/accelerate/config.yaml
```

or, with a config file:

```yaml
# dpo_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
dataset_name: anthropic/hh-rlhf
accelerate_config: zero2  # or path/to/my/accelerate/config.yaml
```

Launch with:

```bash
trl dpo --config dpo_config.yaml
```

</hfoption>
<hfoption id="Reward">

```bash
trl reward \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name trl-lib/ultrafeedback_binarized \
  --accelerate_config zero2  # or path/to/my/accelerate/config.yaml
```

or, with a config file:

```yaml
# reward_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
dataset_name: trl-lib/ultrafeedback_binarized
accelerate_config: zero2  # or path/to/my/accelerate/config.yaml
```

Launch with:

```bash
trl reward --config reward_config.yaml
```

</hfoption>
<hfoption id="GRPO">

```bash
trl grpo \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name HuggingFaceH4/Polaris-Dataset-53K \
  --reward_funcs accuracy_reward \
  --accelerate_config zero2  # or path/to/my/accelerate/config.yaml
```

or, with a config file:

```yaml
# grpo_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
dataset_name: HuggingFaceH4/Polaris-Dataset-53K
reward_funcs:
  - accuracy_reward
accelerate_config: zero2  # or path/to/my/accelerate/config.yaml
```

Launch with:

```bash
trl grpo --config grpo_config.yaml
```

</hfoption>
<hfoption id="RLOO">

```bash
trl rloo \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name HuggingFaceH4/Polaris-Dataset-53K \
  --reward_funcs accuracy_reward \
  --accelerate_config zero2  # or path/to/my/accelerate/config.yaml
```

or, with a config file:

```yaml
# rloo_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
dataset_name: HuggingFaceH4/Polaris-Dataset-53K
reward_funcs:
  - accuracy_reward
accelerate_config: zero2  # or path/to/my/accelerate/config.yaml
```

Launch with:

```bash
trl rloo --config rloo_config.yaml
```

</hfoption>
<hfoption id="KTO">

```bash
trl kto \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name trl-lib/kto-mix-14k \
  --accelerate_config zero2  # or path/to/my/accelerate/config.yaml
```

or, with a config file:

```yaml
# kto_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
dataset_name: trl-lib/kto-mix-14k
accelerate_config: zero2  # or path/to/my/accelerate/config.yaml
```

Launch with:

```bash
trl kto --config kto_config.yaml
```

</hfoption>
</hfoptions>

### Using dataset mixtures

You can use dataset mixtures to combine multiple datasets into a single training dataset. This is useful for training on diverse data sources or when you want to mix different types of data.

<hfoptions id="trainer">
<hfoption id="SFT">

```yaml
# sft_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
datasets:
  - path: stanfordnlp/imdb
  - path: roneneldan/TinyStories
```

Launch with:

```bash
trl sft --config sft_config.yaml
```

</hfoption>
<hfoption id="DPO">

```yaml
# dpo_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
datasets:
  - path: BAAI/Infinity-Preference
  - path: argilla/Capybara-Preferences
```

Launch with:

```bash
trl dpo --config dpo_config.yaml
```

</hfoption>
<hfoption id="Reward">

```yaml
# reward_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
datasets:
  - path: trl-lib/tldr-preference
  - path: trl-lib/lm-human-preferences-sentiment
```

Launch with:

```bash
trl reward --config reward_config.yaml
```

</hfoption>
<hfoption id="GRPO">

```yaml
# grpo_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
datasets:
  - path: HuggingFaceH4/Polaris-Dataset-53K
  - path: trl-lib/DeepMath-103K
reward_funcs:
  - accuracy_reward
```

Launch with:

```bash
trl grpo --config grpo_config.yaml
```

</hfoption>
<hfoption id="RLOO">

```yaml
# rloo_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
datasets:
  - path: HuggingFaceH4/Polaris-Dataset-53K
  - path: trl-lib/DeepMath-103K
reward_funcs:
  - accuracy_reward
```

Launch with:

```bash
trl rloo --config rloo_config.yaml
```

</hfoption>
<hfoption id="KTO">

```yaml
# kto_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
datasets:
  - path: trl-lib/kto-mix-14k
  - path: argilla/ultrafeedback-binarized-preferences-cleaned
```

Launch with:

```bash
trl kto --config kto_config.yaml
```

</hfoption>
</hfoptions>

To see all the available keywords for defining dataset mixtures, refer to the [`scripts.utils.DatasetConfig`] and [`DatasetMixtureConfig`] classes.

## Getting the System Information

You can get the system information by running the following command:

```bash
trl env
```

This will print out the system information, including the GPU information, the CUDA version, the PyTorch version, the transformers version, the TRL version, and any optional dependencies that are installed.

```txt
Copy-paste the following information when reporting an issue:

- Platform: Linux-5.15.0-1048-aws-x86_64-with-glibc2.31
- Python version: 3.11.9
- PyTorch version: 2.4.1
- accelerator(s): NVIDIA H100 80GB HBM3
- Transformers version: 4.45.0.dev0
- Accelerate version: 0.34.2
- Accelerate config: 
  - compute_environment: LOCAL_MACHINE
  - distributed_type: DEEPSPEED
  - mixed_precision: no
  - use_cpu: False
  - debug: False
  - num_processes: 4
  - machine_rank: 0
  - num_machines: 1
  - rdzv_backend: static
  - same_network: True
  - main_training_function: main
  - enable_cpu_affinity: False
  - deepspeed_config: {'gradient_accumulation_steps': 4, 'offload_optimizer_device': 'none', 'offload_param_device': 'none', 'zero3_init_flag': False, 'zero_stage': 2}
  - downcast_bf16: no
  - tpu_use_cluster: False
  - tpu_use_sudo: False
  - tpu_env: []
- Datasets version: 3.0.0
- HF Hub version: 0.24.7
- TRL version: 0.12.0.dev0+acb4d70
- bitsandbytes version: 0.41.1
- DeepSpeed version: 0.15.1
- Diffusers version: 0.30.3
- Liger-Kernel version: 0.3.0
- LLM-Blender version: 0.0.2
- OpenAI version: 1.46.0
- PEFT version: 0.12.0
- vLLM version: not installed
```

This information is required when reporting an issue.
