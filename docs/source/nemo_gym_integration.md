# NeMo-Gym Integration

NVIDIA NeMo-Gym is a library for building reinforcement learning environments for large language models. This integration enables training models in NeMo-Gym environments using TRL's [`GRPOTrainer`].

NeMo-Gym orchestrates multi-step and multi-turn rollouts, providing token IDs and log probabilities to TRL through a custom rollout function. This integration currently requires TRL's vLLM server mode.

## Overview

The integration supports:

- **NeMo-Gym RL environments**: Any NeMo-Gym environment should work through this integration, though not all have been tested. We thorougly tested the following environments in the development of this integration: workplace assistant, reasoning gym, mcqa, and math with judge.
- **Multi-turn tasks**: Multi-step environments involve the agent performing multiple tool calls or other steps sequentially. Multi-turn environments involve follow-up user messages, in addition to potentially multiple tool calls or other steps in the environment.
- **Multi-environment training**: Train on multiple tasks or environments simultaneously and efficiently at scale.

## Why NeMo Gym

NeMo-Gym was designed to support large-scale, production-grade reinforcement learning training:

- **Scale and Coverage**: NeMo-Gym supports diverse environments running in parallel, with many examples across domains (math, coding, tool use, knowledge, reasoning, search, ...). 
- **Production-Ready**: Tested for frontier model training at large scale. The infrastructure is designed for the scale and reliability required for production LLM training.
- **Multi-Verifier RL Training**: Built for training with multiple verification methods simultaneously. Supports algorithmic verification (code execution, math verification), LLM-as-a-judge, and custom verification logic across different environments in a single training run.
- **Decoupled Architecture**: Enables building agents and environments independently from the training loop. Environments can be developed, tested, and deployed without requiring expertise in the RL training framework.
- **OpenAI-Compatible API**: All environments are compatble with standardized OpenAI Responses API, allowing seamless integration with any inference server (vLLM, SGLang, etc.) and enabling environment reuse across different training frameworks.
- **Container-Ready**: Designed for containerized deployment with REST APIs, supporting complex multi-agent systems and environments like SWE-Bench that require isolated Docker containers.

## Installation

Install TRL with vLLM support:

```bash
pip install trl[vllm]
```

Install NeMo-Gym:

```bash
git clone https://github.com/NVIDIA-NeMo/Gym.git
cd Gym
uv venv --python 3.12
source .venv/bin/activate
uv sync --extra dev
```

## Available Environments

NeMo-Gym provides training-ready environments across various domains, including but not limited to:

| Environment | Domain | Description |
|-------------|--------|-------------|
| Workplace Assistant | Agent | Multi-step tool calling in common office scenarios (calendar, email, etc.) |
| Math with Judge | Math | Math problems with algorithmic or judge-based verification |
| Code Gen | Coding | Competitive programming problems with code execution |
| MCQA | Knowledge | Multiple-choice question answering |
| Instruction Following | Instruction Following | IFEval/IFBench style tasks |
| Reasoning Gym | Multiple | Single-step procedurally generated verifiable tasks across various domains |

See a complete list of available training environments in the [NeMo-Gym repository](https://github.com/NVIDIA-NeMo/Gym#-available-resource-servers).

## Preparing a Dataset

For creating a new environment, check out the [official guide](https://docs.nvidia.com/nemo/gym/latest/contribute/environments/new-environment.html).

Many NeMo-Gym datasets used in training Nemotron models are available on Hugging Face, corresponding to existing RL environments. 

### Download and Prepare Workplace Assistant Data

Use `ng_prepare_data` to download and prepare the dataset. This command:
- Downloads the dataset from Hugging Face
- Validates the data format
- Adds an `agent_ref` field to each example that tells NeMo-Gym which agent server should handle that example

Note that `train_multi_env.py` adds `agent_ref` field when loading datasets in case that datasets are created some other way.

First, set `env.yaml` in `Gym/` to contain your Hugging Face token: 
```
hf_token: <your_hf_token>
```

Example dataset preparation for the workplace assistant environment:

```bash
cd Gym
source .venv/bin/activate

config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
resources_servers/workplace_assistant/configs/workplace_assistant.yaml"

ng_prepare_data "+config_paths=[${config_paths}]" \
    +output_dirpath=data/workplace_assistant \
    +mode=train_preparation \
    +should_download=true \
    +data_source=huggingface
```

This creates `train.jsonl` and `validation.jsonl` files in `data/workplace_assistant/`. 

### Dataset Format

In NeMo Gym, datasets are stored as JSONL. Each line contains a task with input messages, potential tool definitions, metadata such as ground truth for verification, and an agent server reference. The workplace dataset is structured like shown below. The metadata fields can differ between datasets, as long as the corresponding resources server leverages the fields appropriately.

```json
{
  "responses_create_params": {
    "input": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "Move any of jinsoo's tasks that are in review to completed"}
    ],
    "tools": [...],  // Full tool definitions
    "parallel_tool_calls": false,
    "temperature": 1
  },
  "ground_truth": [
    {"name": "project_management_update_task", "arguments": "{...}"},
    ...
  ],
  "category": "workbench_project_management",
  "environment_name": "workbench",
  "agent_ref": {
    "type": "responses_api_agents",
    "name": "workplace_assistant_simple_agent"
  }
}
```

## Training Configuration

Create a `config_workplace.yaml` file with your training parameters:

```yaml
model_name: "Qwen/Qwen2.5-1.5B-Instruct"

dataset_path: "data/workplace_assistant/train.jsonl"
eval_dataset_path: "data/workplace_assistant/validation.jsonl"

task: 'workplace'               # used in wandb run name
output_dir: "outputs/nemo_gym"
report_to: "wandb"              # set to none if you don't have wandb set up.
project_name: "trl-nemo-gym"

learning_rate: 1.0e-5
max_steps: 1000
num_generations: 8
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
max_seq_length: 16384

temperature: 1.0
top_p: 0.999

save_steps: 10
eval_strategy: "steps"
eval_steps: 10
```

## Interactive Training

For development and testing on a single node:

### Step 1: Update environment config

Update `env.yaml` to include model information: 

```
policy_base_url: http://127.0.0.1:8000/v1
policy_api_key: EMPTY
policy_model_name: Qwen/Qwen3-30B-A3B-Instruct-2507
hf_token: ...
```

### Step 2: Start NeMo-Gym Servers

First, start the NeMo-Gym environment servers:

```bash
cd Gym
source .venv/bin/activate

config_paths="resources_servers/workplace_assistant/configs/workplace_assistant.yaml,\
responses_api_models/vllm_model/configs/vllm_model_for_training.yaml"

ng_run "+config_paths=[${config_paths}]"
```

This starts:
- **Head server**: Manages servers used in training
- **Agent server**: Orchestrates rollouts by leveraging resource servers and model servers
- **Resources server**: Supports environment logic such as state-based feedback, tool implementations, and task verification
- **Model server**: Adapts vLLM server requests to support NeMo Gym agents and ensures OpenAI API compatibility

### Step 2: Start TRL vLLM Server

In a second terminal, start the TRL vLLM server on GPU 0:

```bash
cd trl

CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --max-model-len 16384 \
  --host 0.0.0.0 \
  --port 8000
```


### Step 3: Run Training

In a third terminal, launch the training script on GPU 1:

```bash
cd trl/
source .venv/bin/activate

cd examples/scripts/nemo_gym

# if using wandb
export WANDB_API_KEY=...
uv pip install wandb       # TODO: double check its missing from trl 

CUDA_VISIBLE_DEVICES=1 python train_multi_env.py --config config_workplace.yaml
```

Note that these separate terminals can also be tmux sessions or processes ran in the background.

## Multi-Node Training with Slurm

An example 5-node training script is provided in `submit.sh`. To use this, update your slurm account and partition, path to Gym repository, and your training config.

Nodes 1-4 are used for training backend, while node 5 is used for vLLM inference. For more details on TRL's vLLM integration, visit vllm integration page.  

Submit the job:
```bash
sbatch submit.sh
```

Monitor training logs:
```bash
tail -f logs/<job_id>/*
```

Set up wandb logging for detailed training metrics!

## Multi-Environment Training

Train on multiple NeMo-Gym environments simultaneously. This allows learning diverse capabilities (e.g., tool calling + math reasoning) in a single training run.

### Step 1: Prepare Individual Datasets

First, prepare datasets for each environment you want to use. Above, we prepared the workplace dataset. Now, create a reasoning gym dataset: 

```bash
cd Gym
source .venv/bin/activate
uv add reasoning-gym
cd resources_servers/reasoning_gym
python scripts/create_dataset.py \
    --task mini_sudoku \
    --size 2000 \
    --seed 42 \
    --output data/reasoning_gym/train_mini_sudoku.jsonl

python scripts/create_dataset.py \
    --task mini_sudoku \
    --size 50 \
    --seed 24 \
    --output data/reasoning_gym/val_mini_sudoku.jsonl
```

### Step 2: Create Blended Dataset

Create a single dataset with tasks from both environments mixed together. This can be done with a simple bash command, such as the following: 
```bash
cat data/workplace_assistant/train_workplace.jsonl data/reasoning_gym/train_mini_sudoku.jsonl | shuf > train_multi_env.jsonl
```

Note you may want to ensure that the datasets are the same size before shuffling to get an even blend of tasks. Do the same for the validation dataset.

### Step 3: Update Training Config

Create `config_multi_env.yaml` pointing to the blended dataset:

```yaml
model_name: "Qwen/Qwen3-4B-Instruct-2507"

dataset_path: "/path/to/data/train_multi_env.jsonl"
eval_dataset_path: "/path/to/data/val_multi_env.jsonl"

task: "workplace-sudoku"                    # used in wandb run name
output_dir: "outputs/nemo_gym_multi_env"

# ... rest of config same
```

### Step 4: Launch Resources Servers

Start NeMo-Gym with both resources servers in the config:

```bash
cd Gym
source .venv/bin/activate

config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
resources_servers/workplace_assistant/configs/workplace_assistant.yaml,\
resources_servers/reasoning_gym/configs/reasoning_gym.yaml"

ng_run "+config_paths=[${config_paths}]" +head_server.host=0.0.0.0
```

This starts servers for both environments. The training script will automatically route each example to the correct agent server based on its `agent_ref` field.

### Step 5: Run Training

Just update the slurm submission script to use the new config, then submit the job as before!

The training script reads `agent_ref` from each example's metadata, routes requests to the correct NeMo-Gym agent server, and handles different environments in the same batch

## Resources

- [NeMo-Gym GitHub](https://github.com/NVIDIA-NeMo/Gym)
- [NeMo-Gym Documentation](https://docs.nvidia.com/nemo/gym/latest/)
- [Training Script](https://github.com/huggingface/trl/blob/main/examples/scripts/nemo_gym/train_multi_env.py)
- [TRL GRPO Trainer](grpo_trainer)