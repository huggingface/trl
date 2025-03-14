# Command Line Interfaces (CLIs)

You can use TRL to fine-tune your language model with methods like Supervised Fine-Tuning (SFT) or Direct Policy Optimization (DPO) using the command line interface (CLI).

Currently supported CLIs are:

#### Training commands

- `trl dpo`: fine-tune a LLM with DPO
- `trl grpo`: fine-tune a LLM with GRPO
- `trl kto`: fine-tune a LLM with KTO
- `trl sft`: fine-tune a LLM with SFT

#### Other commands

- `trl env`: get the system information

## Fine-tuning with the CLI

Before getting started, pick up a Language Model from Hugging Face Hub. Supported models can be found with the filter "text-generation" within models. Also make sure to pick up a relevant dataset for your task.

Before using the `sft` or `dpo` commands make sure to run:
```bash
accelerate config
```
and pick up the right configuration for your training setup (single / multi-GPU, DeepSpeed, etc.). Make sure to complete all steps of `accelerate config` before running any CLI command.

We also recommend you passing a YAML config file to configure your training protocol. Below is a simple example of a YAML file that you can use for training your models with `trl sft` command.

```yaml
model_name_or_path:
  Qwen/Qwen2.5-0.5B
dataset_name:
  stanfordnlp/imdb
report_to:
  none
learning_rate:
  0.0001
lr_scheduler_type:
  cosine
```

Save that config in a `.yaml` and get started immediately! An example CLI config is available as `examples/cli_configs/example_config.yaml`. Note you can overwrite the arguments from the config file by explicitly passing them to the CLI, e.g. from the root folder:

```bash
trl sft --config examples/cli_configs/example_config.yaml --output_dir test-trl-cli --lr_scheduler_type cosine_with_restarts
```

Will force-use `cosine_with_restarts` for `lr_scheduler_type`.

### Supported Arguments 

We do support all arguments from `transformers.TrainingArguments`, for loading your model, we support all arguments from `~trl.ModelConfig`:

[[autodoc]] ModelConfig

You can pass any of these arguments either to the CLI or the YAML file.

### Supervised Fine-tuning (SFT)

Follow the basic instructions above and run `trl sft --output_dir <output_dir> <*args>`: 

```bash
trl sft --model_name_or_path facebook/opt-125m --dataset_name stanfordnlp/imdb --output_dir opt-sft-imdb
```

The SFT CLI is based on the `trl/scripts/sft.py` script.

### Direct Policy Optimization (DPO)

To use the DPO CLI, you need to have a dataset in the TRL format such as 

* TRL's Anthropic HH dataset: https://huggingface.co/datasets/trl-internal-testing/hh-rlhf-helpful-base-trl-style
* TRL's OpenAI TL;DR summarization dataset: https://huggingface.co/datasets/trl-internal-testing/tldr-preference-trl-style

These datasets always have at least three columns `prompt, chosen, rejected`:

* `prompt` is a list of strings.
* `chosen` is the chosen response in [chat format](https://huggingface.co/docs/transformers/main/en/chat_templating)
* `rejected` is the rejected response [chat format](https://huggingface.co/docs/transformers/main/en/chat_templating) 


To do a quick start, you can run the following command:

```bash
trl dpo --model_name_or_path facebook/opt-125m --output_dir trl-hh-rlhf --dataset_name trl-internal-testing/hh-rlhf-helpful-base-trl-style
```


The DPO CLI is based on the `trl/scripts/dpo.py` script.


#### Custom preference dataset

Format the dataset into TRL format (you can adapt the `examples/datasets/anthropic_hh.py`):

```bash
python examples/datasets/anthropic_hh.py --push_to_hub --hf_entity your-hf-org
```

## Chat interface

<Tip warning={true}>

The chat interface is deprecated and will be removed in TRL 0.19. Use `transformers-cli chat` instead. For more information, see the [Transformers documentation, chat with text generation models](https://huggingface.co/docs/transformers/quicktour#chat-with-text-generation-models).

</Tip>

The chat CLI lets you quickly load the model and talk to it. Simply run the following:

<pre><code>$ trl chat --model_name_or_path Qwen/Qwen1.5-0.5B-Chat 
<strong><span style="color: red;">&lt;quentin_gallouedec&gt;:</span></strong>
What is the best programming language?

<strong><span style="color: blue;">&lt;Qwen/Qwen1.5-0.5B-Chat&gt;:</span></strong>
There isn't a "best" programming language, as everyone has different style preferences, needs, and preferences. However, some people commonly use   
languages like Python, Java, C++, and JavaScript, which are popular among developers for a variety of reasons, including readability, flexibility,  
and scalability. Ultimately, it depends on personal preference, needs, and goals.
</code></pre>

Note that the chat interface relies on the tokenizer's [chat template](https://huggingface.co/docs/transformers/chat_templating) to format the inputs for the model. Make sure your tokenizer has a chat template defined.

Besides talking to the model there are a few commands you can use:

- `clear`: clears the current conversation and start a new one
- `example {NAME}`: load example named `{NAME}` from the config and use it as the user input
- `set {SETTING_NAME}={SETTING_VALUE};`: change the system prompt or generation settings (multiple settings are separated by a `;`).
- `reset`: same as clear but also resets the generation configs to defaults if they have been changed by `set`
- `save` or `save {SAVE_NAME}`: save the current chat and settings to file by default to `./chat_history/{MODEL_NAME}/chat_{DATETIME}.yaml` or `{SAVE_NAME}` if provided
- `exit`: closes the interface

## Getting the system information

You can get the system information by running the following command:

```bash
trl env
```

This will print out the system information including the GPU information, the CUDA version, the PyTorch version, the transformers version, and the TRL version, and any optional dependencies that are installed.

```txt
Copy-paste the following information when reporting an issue:

- Platform: Linux-5.15.0-1048-aws-x86_64-with-glibc2.31
- Python version: 3.11.9
- PyTorch version: 2.4.1
- CUDA device: NVIDIA H100 80GB HBM3
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
```

This information are required when reporting an issue.
