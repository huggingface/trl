# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl[vllm]",
#     "nemo_gym @ git+https://github.com/NVIDIA-NeMo/Gym",
# ]
# ///

import argparse
import os
from dataclasses import dataclass

import yaml
from nemo_gym.rollout_collection import RolloutCollectionHelper
from nemo_gym.server_utils import BaseServerConfig
from nemo_gym_utils import launch_nemo_gym, load_dataset_from_jsonl, nemo_gym_rollout_func, reward_fn

from trl import GRPOConfig, GRPOTrainer


@dataclass
class NeMoGymGRPOConfig(GRPOConfig):
    rch: RolloutCollectionHelper | None = None
    head_server_config: BaseServerConfig | None = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="examples/scripts/nemo_gym/config.yaml", help="Path to config file")
    parser.add_argument("--vllm_server_host", type=str, default="127.0.0.1")
    parser.add_argument("--head_server_host", type=str, default="127.0.0.1")
    parser.add_argument("--head_server_port", type=int, default=11000)
    parser.add_argument("--vllm_server_port", type=int, default=8000)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_name = config.pop("model_name")
    dataset_path = config.pop("dataset_path")
    eval_dataset_path = config.pop("eval_dataset_path", None)
    gym_configs = config.pop("gym_configs", None)
    task = config.pop("task", None)
    project_name = config.pop("project_name", None)

    if "learning_rate" in config and isinstance(config["learning_rate"], str):
        config["learning_rate"] = float(config["learning_rate"])
    if "weight_decay" in config and isinstance(config["weight_decay"], str):
        config["weight_decay"] = float(config["weight_decay"])

    run_helper = None
    is_rank_zero = int(os.environ.get("RANK", "0")) == 0
    if gym_configs and is_rank_zero:
        run_helper = launch_nemo_gym(
            config_paths=gym_configs,
            model_name=model_name,
            vllm_server_host=args.vllm_server_host,
            vllm_server_port=args.vllm_server_port,
            host="0.0.0.0",
            port=args.head_server_port,
        )

    head_server_config = BaseServerConfig(host=args.head_server_host, port=args.head_server_port)
    rch = RolloutCollectionHelper()

    if project_name:
        os.environ["WANDB_PROJECT"] = project_name

    dataset = load_dataset_from_jsonl(dataset_path)

    eval_dataset = None
    if eval_dataset_path:
        eval_dataset = load_dataset_from_jsonl(eval_dataset_path)

    training_args = NeMoGymGRPOConfig(
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host=args.vllm_server_host,
        vllm_server_port=args.vllm_server_port,
        gradient_checkpointing=True,
        num_generations_eval=1,
        logging_steps=1,
        epsilon=0.2,
        epsilon_high=0.28,
        loss_type="grpo",
        mask_truncated_completions=True,
        shuffle_dataset=False,
        model_init_kwargs={"torch_dtype": "auto"},
        rch=rch,
        head_server_config=head_server_config,
        eval_on_start=True,
        ddp_timeout=7200,
        **config,
    )

    if training_args.run_name is None:
        task_name = task or os.path.basename(dataset_path).replace(".jsonl", "").replace(".json", "")
        model_short = model_name.split("/")[-1]
        is_corr = getattr(training_args, "vllm_importance_sampling_correction", None)
        training_args.run_name = (
            f"{task_name}_{model_short}"
            f"_rpp{training_args.num_generations}"
            f"_dbs{training_args.per_device_train_batch_size}"
            f"_ga{training_args.gradient_accumulation_steps}"
            f"{f'_IS{str(bool(is_corr)).lower()}' if is_corr is not None else ''}"
            f"_maxlen{training_args.max_completion_length}"
            f"_lr{training_args.learning_rate}"
            f"_temp{training_args.temperature}"
            f"_topp{training_args.top_p}"
        )

    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=reward_fn,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        rollout_func=nemo_gym_rollout_func,
        args=training_args,
    )

    try:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    finally:
        if run_helper is not None:
            run_helper.shutdown()


if __name__ == "__main__":
    main()
