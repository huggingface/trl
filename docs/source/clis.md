# Command Line Interfaces (CLIs)

TRL provides a powerful command-line interface (CLI) to fine-tune large language models (LLMs) using methods like Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and more. The CLI abstracts away much of the boilerplate, letting you launch training jobs quickly and reproducibly.

Currently supported commands are:

#### Training commands

- `trl dpo`: fine-tune a LLM with DPO
- `trl grpo`: fine-tune a LLM with GRPO
- `trl kto`: fine-tune a LLM with KTO
- `trl sft`: fine-tune a LLM with SFT

#### Other commands

- `trl env`: get the system information
- `trl vllm-serve`: serve a model with vLLM

## Fine-tuning with the CLI

To fine-tune a model, for example, you can run:

<hfoptions id="command_line">
<hfoption id="sft">

```bash
trl sft --model_name_or_path Qwen/Qwen2.5-0.5B --dataset_name stanfordnlp/imdb
```

</hfoption>
<hfoption id="dpo">

```bash
trl dpo --model_name_or_path Qwen/Qwen2.5-0.5B --dataset_name ...
```

</hfoption>
</hfoptions>

### Configuration file

You can also configure your training setup using a YAML configuration file, which helps keep your command-line usage clean and reproducible. Below is an example of a minimal configuration file:

<hfoptions id="config_file">
<hfoption id="sft">

```yaml
# example_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
dataset_name: stanfordnlp/imdb
```

To launch training with this config, run:

```bash
trl sft --config example_config.yaml
```

</hfoption>
<hfoption id="dpo">

```yaml
# example_config.yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
dataset_name: ...
```

To launch training with this config, run:

```bash
trl dpo --config example_config.yaml
```

</hfoption>
</hfoptions>

### Use the CLI for distributed training

The TRL CLI supports **all the arguments** of `accelerate launch`. See https://huggingface.co/docs/accelerate/en/basic_tutorials/launch#using-accelerate-launch. Consequelntly you can easily distribute the training leveraging `accelerate`. Example with `num_processes`:


<hfoptions id="launch_args">
<hfoption id="sft">

```bash
trl sft --model_name_or_path Qwen/Qwen2.5-0.5B --dataset_name stanfordnlp/imdb --num_processes 4
```

</hfoption>
<hfoption id="dpo">

```bash
trl dpo --model_name_or_path Qwen/Qwen2.5-0.5B --dataset_name ... --num_processes 4
```

</hfoption>
</hfoptions>

TRL provides some predefined configurations for distrubtued training. To use then  simply use the `--accelerate_config` argument. For example, to use the DeepSpeed ZeRO Stage 2, run:

<hfoptions id="predefined_configs">
<hfoption id="sft">

```bash
trl sft --model_name_or_path Qwen/Qwen2.5-0.5B --dataset_name stanfordnlp/imdb --accelerate_config deepspeed_zero2
```

</hfoption>
<hfoption id="dpo">

```bash
trl dpo --model_name_or_path Qwen/Qwen2.5-0.5B --dataset_name ... --accelerate_config deepspeed_zero2
```

</hfoption>
</hfoptions>

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
