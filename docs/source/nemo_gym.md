# NeMo Gym Integration

NVIDIA NeMo Gym is a library for building RL environments for large language models. This integration enables training models in NeMo Gym environments using TRL's GRPOTrainer with vLLM server mode.

The integration supports multi-step and multi-turn rollouts, multi-environment training, and any NeMo Gym environment (thoroughly tested: workplace assistant, reasoning gym, MCQA, and math with judge).

## Why NeMo Gym

- **Production-Ready Scale**: Tested for frontier model training with diverse environments running in parallel across math, coding, tool use, reasoning, and more.
- **Multi-Verifier Training**: Supports algorithmic verification, LLM-as-a-judge, and custom verification logic in a single training run.
- **Decoupled Architecture**: Build agents and environments independently from the training loop—no RL framework expertise required.
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

For a complete list of available training environments, refer to the [NeMo Gym repository](https://github.com/NVIDIA-NeMo/Gym#-available-resource-servers).

## Before You Start

Complete these one-time setup steps before running training.

### Install TRL and NeMo Gym

1. **Install TRL with vLLM extras**

   ```bash
   cd trl/
   uv venv
   source .venv/bin/activate
   uv sync --extra vllm
   ```

1. **Install NeMo Gym**

   ```bash
   # deactivate trl venv
   deactivate
   git clone https://github.com/NVIDIA-NeMo/Gym.git
   cd Gym
   uv venv --python 3.12
   source .venv/bin/activate
   uv sync
   ```

### Prepare a Dataset

Many NeMo Gym datasets used to train Nemotron models are available on Hugging Face. Use `ng_prepare_data` to download and prepare datasets. This command:

- Downloads the dataset from Hugging Face
- Validates the data format
- Adds an `agent_ref` field to each example that tells NeMo Gym which agent server should handle that example

> **Note**: `train_multi_environment.py` adds the `agent_ref` field when loading datasets, so this step is optional if datasets are created another way.

1. **Set Hugging Face Token**

   Create `env.yaml` in `Gym/` with your HF token:

   ```yaml
   hf_token: <your_hf_token>
   ```

1. **Prepare Dataset**

   ```bash
   # Enter Gym and activate the venv
   cd Gym
   source .venv/bin/activate

   # Set config paths
   config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
   resources_servers/workplace_assistant/configs/workplace_assistant.yaml"

   # Download data and prep for training
   ng_prepare_data "+config_paths=[${config_paths}]" \
       +output_dirpath=data/workplace_assistant \
       +mode=train_preparation \
       +should_download=true \
       +data_source=huggingface
   ```

   This creates `train.jsonl` and `validation.jsonl` files in `data/workplace_assistant/`.

To create a new environment, refer to the [environment creation guide](https://docs.nvidia.com/nemo/gym/latest/contribute/environments/new-environment.html). We suggest running an existing one first!

#### Dataset Format

NeMo Gym datasets are stored as JSONL. Each line contains a task with input messages, tool definitions, metadata such as ground truth for verification, and an agent server reference. The following example shows the workplace dataset structure. Metadata fields can differ between datasets, as long as the corresponding resources server uses the fields appropriately.

```json
{
  "responses_create_params": {
    "input": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "Move any of jinsoo's tasks that are in review to completed"}
    ],
    "tools": [...],
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

## Interactive Training

For development and testing on a single node.

### Set Up

1. **Update Environment Config**

   Update `env.yaml` in `Gym/` to include model information:

   ```yaml
   policy_base_url: http://127.0.0.1:8000/v1
   policy_api_key: EMPTY
   policy_model_name: Qwen/Qwen2.5-1.5B-Instruct
   hf_token: ...
   ```

2. **Update Training Config**

   Update `examples/scripts/nemo_gym/config.yaml` to point to the dataset generated above, and any other optional modifications.

###  Run Training

The following steps run in 3 terminals. It can also be ran with processes in the background, or using tmux.

1. **Start NeMo Gym Servers** (Terminal 1)

   ```bash
   cd Gym/
   source .venv/bin/activate

   config_paths="resources_servers/workplace_assistant/configs/workplace_assistant.yaml,\
   responses_api_models/vllm_model/configs/vllm_model_for_training.yaml"

   ng_run "+config_paths=[${config_paths}]"
   ```

   This starts:
   - **Agent server**: Orchestrates rollouts using resource servers and model servers
   - **Resources server**: Supports environment logic such as state-management, tool implementations, and task verification
   - **Model server**: Adapts vLLM server requests to support NeMo Gym agents and on-policy RL training while ensuring OpenAI API compatibility
   - **Head server**: Manages servers used in training enabling their discovery

1. **Start TRL vLLM Server on GPU 0** (Terminal 2)

   ```bash
   cd trl/
   source .venv/bin/activate
   CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
     --model Qwen/Qwen2.5-1.5B-Instruct \
     --max-model-len 16384 \
     --host 0.0.0.0 \
     --port 8000
   ```

1. **Run Training on GPU 1** (Terminal 3)

   ```bash
   source trl/.venv/bin/activate
   cd trl/examples/scripts/nemo_gym
   export WANDB_API_KEY=... 
   uv add omegaconf 

   CUDA_VISIBLE_DEVICES=1 python train_multi_environment.py --config config.yaml
   ```

## Multi-Node Training with Slurm

An example five-node training script is provided in `submit.sh`. Nodes one through four run the training algorithm, while node five runs vLLM inference for NeMo Gym agent rollouts.

1. **Configure the Script**

   Update `submit.sh` with your Slurm account, partition, paths to your project directory, and updated training configs.

1. **Submit the Job**

   ```bash
   sbatch submit.sh
   ```

1. **Monitor Training**

   ```bash
   tail -f logs/<job_id>/*
   ```

> **Tip**: Set up wandb logging for detailed training metrics. For more details on TRL's vLLM integration, refer to the vLLM integration page.

## Multi-Environment Training

Train on multiple NeMo Gym environments simultaneously. This allows learning diverse capabilities, such as tool calling and math reasoning, in a single training run.

1. **Prepare Individual Datasets**

   Prepare datasets for each environment. The workplace assistant dataset was prepared above. Now lets create a dataset for the mini sudoku environment implemented by the reasoning gym resources server in NeMo Gym:

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

1. **Create Combined Dataset**

   Combine datasets into a single file with tasks from both environments:

   ```bash
   cat data/workplace_assistant/train_workplace.jsonl data/reasoning_gym/train_mini_sudoku.jsonl | shuf > train_multi_env.jsonl
   ```

   > **Tip**: Ensure datasets are the same size before shuffling for an even blend of tasks. Repeat for the validation dataset.

1. **Update Training Config**

   Update the config to point to the combined dataset:

   ```yaml
   model_name: "Qwen/Qwen3-4B-Instruct-2507"

   dataset_path: "/path/to/data/train_multi_env.jsonl"
   eval_dataset_path: "/path/to/data/val_multi_env.jsonl"

   task: "workplace-sudoku"                    # used in wandb run name
   output_dir: "outputs/nemo_gym_multi_env"

   # ... rest of config same
   ```

1. **Update ng_run**

   Whether training interactively or via Slurm, update the `ng_run` command to include config files from each resources server:

   ```bash
   cd Gym
   source .venv/bin/activate

   config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
   resources_servers/workplace_assistant/configs/workplace_assistant.yaml,\
   resources_servers/reasoning_gym/configs/reasoning_gym.yaml"

   ng_run "+config_paths=[${config_paths}]"
   ```

   This starts servers for both environments. The training script automatically routes each example to the correct agent server based on its `agent_ref` field.

1. **Run Training**

   Update the Slurm submission script to use the new training config and both `ng_run` resources server configs, then submit the job as before.

   The training script reads `agent_ref` from each example's metadata, routes requests to the correct NeMo Gym agent server, and handles different agents and environments in the same batch.

## Resources

- [NeMo Gym GitHub](https://github.com/NVIDIA-NeMo/Gym)
- [NeMo Gym Documentation](https://docs.nvidia.com/nemo/gym/latest/)
- [Training Script](https://github.com/huggingface/trl/blob/main/examples/scripts/nemo_gym/train_multi_environment.py)
- [TRL GRPO Trainer](grpo_trainer)
