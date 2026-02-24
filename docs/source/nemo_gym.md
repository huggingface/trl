# NeMo Gym Integration

NVIDIA NeMo Gym is a library for building RL environments for large language models. This integration enables training models in NeMo Gym environments using TRL's GRPOTrainer. Multi-turn and multi-environment training are both supported!

Note that a minimum of 2 GPUs is currently required, as this integration relies on TRL's vLLM server mode.  

## Why NeMo Gym

- **Tested at scale**: Battle-tested RL infra used in Nemotron post-training.
- **Multi-environment training**: Supports parallel training of complex agents in diverse environments, such as coding agents, deep research, workplace tasks, math, science, and more.
- **Decoupled architecture**: Build agents and environments independently from the training loop, no RL framework expertise required.
- **OpenAI-Compatible API**: All environments use the standardized OpenAI Responses API for seamless integration with vLLM, OpenAI models, and other endpoints.

## Available Environments

NeMo Gym provides training-ready environments across multiple domains, including but not limited to:

| Environment | Domain | Description |
|-------------|--------|-------------|
| Workplace Assistant | Agent | Multi-step tool calling in common office scenarios (calendar, email, and more) |
| Math with Judge | Math | Math problems with algorithmic or judge-based verification |
| Code Gen | Coding | Competitive programming problems with code execution |
| MCQA | Knowledge | Multiple-choice question answering |
| Instruction Following | Instruction Following | IFEval/IFBench style tasks |
| Reasoning Gym | Multiple | Single-step procedurally generated verifiable tasks across domains |

For a complete list of available training environments, refer to the [NeMo Gym repository](https://github.com/NVIDIA-NeMo/Gym).

## Quickstart

First install TRL and NeMo Gym with some extra packages:

### Install TRL and NeMo Gym

```bash
cd trl/
uv venv --python 3.12
source .venv/bin/activate
uv sync --extra vllm
uv pip install fastapi uvicorn accelerate deepspeed wandb omegaconf

git clone https://github.com/NVIDIA-NeMo/Gym
uv pip install -e Gym/
```

### Prepare a Dataset

In this example we will train a model on the workplace assistant environment, a multi-step tool use environment for common office scenarios. The dataset is available on Hugging Face. Use `ng_prepare_data` to download and prepare it:

```bash
cd Gym
echo 'hf_token: YOUR_HF_TOKEN' > env.yaml
ng_prepare_data \
    "+config_paths=[responses_api_models/vllm_model/configs/vllm_model.yaml,resources_servers/workplace_assistant/configs/workplace_assistant.yaml]" \
    +output_dirpath=resources_servers/workplace_assistant/data \
    +mode=train_preparation \
    +should_download=true \
    +data_source=huggingface

```

Make sure you have `train.jsonl` and `validation.jsonl`.

## Interactive Training

### Setup

Update path to the dataset in the config: `examples/scripts/nemo_gym/config.yaml`. 

### Run Training

Training with NeMo Gym and TRL requires vLLM server mode. First, start the vLLM server:

1. **Start TRL vLLM Server on GPU 0**

   ```bash
   CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
     --model Qwen/Qwen2.5-1.5B-Instruct \
     --max-model-len 16384 \
     --host 0.0.0.0 \
     --port 8000
   ```

Now launch training!

1. **Run Training on GPU 1**

   ```bash
   CUDA_VISIBLE_DEVICES=1 python3 examples/scripts/nemo_gym/grpo_nemo_gym.py
   ```

You should see training progress with completions logged to the terminal! Set up WandB or Trackio to monitor detailed metrics. 

## Using other environments

Using other NeMo Gym environments in TRL is simple. First, update `gym_configs` in `config.yaml` to point to the new NeMo Gym config file. Next, download or create a new dataset. Note that NeMo Gym datasets require an `agent_ref` field so that rollouts are generated in the correct environment for each task. Visit the [NeMo Gym documentation](https://docs.nvidia.com/nemo/gym/latest/) to learn more about configuration files, datasets, and creating new NeMo Gym environments.

## Multi-Environment Training

To train on multiple environments simultaneously, create a dataset with tasks from both environments. Add each environment config to the `gym_configs` list in your training config. NeMo Gym automatically routes each example to the correct agent server based on its `agent_ref` field for effortless and scalable multi-environment training.

Visit the NeMo Gym documentation to learn more about existing environments and how to build a new one!

## Multi-Node Training with Slurm

An example Slurm submission script is provided in `submit.sh`. Update it with your Slurm account, partition, and local paths, then submit with `sbatch submit.sh`.

## Resources

- [NeMo Gym GitHub](https://github.com/NVIDIA-NeMo/Gym)
- [NeMo Gym Documentation](https://docs.nvidia.com/nemo/gym/latest/)
- [Training Script](https://github.com/huggingface/trl/blob/main/examples/scripts/nemo_gym/grpo_nemo_gym.py)
- [TRL GRPO Trainer](grpo_trainer)
