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
uv pip install fastapi uvicorn accelerate deepspeed wandb omegaconf reasoning-gym
git clone https://github.com/NVIDIA-NeMo/Gym
uv pip install -e Gym/
```

### Prepare a Dataset

In this example we will train a model to play sudoku. First, generate a dataset: 

```bash
python3 Gym/resources_servers/reasoning_gym/scripts/create_dataset.py \
    --task mini_sudoku \
    --size 2000 \
    --seed 42 \
    --output data/train_mini_sudoku.jsonl

python3 Gym/resources_servers/reasoning_gym/scripts/create_dataset.py \
    --task mini_sudoku \
    --size 50 \
    --seed 24 \
    --output data/val_mini_sudoku.jsonl
```

## Interactive Training

### Setup

Update path to the generated datasets in the config: `examples/scripts/nemo_gym/config.yaml`. 

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

You should see training progress with completions logged to the terminal! Set up WandB or Trackio for detailed metrics. You should see training results like this:

![nemo_gym_sudoku_train](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/nemo_gym_sudoku_train.png)

![nemo_gym_sudoku_eval](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/nemo_gym_sudoku_eval.png)


## Using Other NeMo Gym Environments

Using other NeMo Gym environments in TRL is simple. First, update `gym_configs` in `config.yaml` to point to the new NeMo Gym config file. Next, [download](https://huggingface.co/collections/nvidia/nemo-gym) or create a new dataset. Note that NeMo Gym datasets require an `agent_ref` field so that rollouts are generated in the correct environment for each task. Visit the [NeMo Gym documentation](https://docs.nvidia.com/nemo/gym/latest/) to learn more about configuration files, datasets, and creating new NeMo Gym environments.

## Multi-Environment Training

To train on multiple environments simultaneously, create a dataset with tasks from both environments. Add each environment config to the `gym_configs` list in your training config. NeMo Gym automatically routes each example to the correct agent server based on its `agent_ref` field for effortless and scalable multi-environment training.

Visit the NeMo Gym documentation to learn more about existing environments and how to build a new one!

## Resources

- [NeMo Gym GitHub](https://github.com/NVIDIA-NeMo/Gym)
- [NeMo Gym Documentation](https://docs.nvidia.com/nemo/gym/latest/)
