# Copyright 2025 The HuggingFace Team. All rights reserved.
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

"""

------------------------------------------------------------------------------------------------------------
1 machine       | 4 GPUs for training, 2 GPUS for VLLM      | using NCCL to deliver param updates
------------------------------------------------------------------------------------------------------------

---
(1) start MAIN TRAINING script:
    (4 GPUs for training)
---
    CUDA_VISIBLE_DEVICES='0,1,2,3' \
    accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
        --num_processes=4 \
        grpo_with_remote_vllm.py \
        --model_name_or_path /mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-7B-Instruct/ \
        --dataset_name "trl-internal-testing/zen" \
        --output_dir './mytests' \
        --bf16 \
        --use_remote_vllm=True --vllm_max_model_len 4096 --remote_vllm_num_gpus=2
---
(2) start VLLM script (do not run the commandline below, it's only a demo, the true commandline will be `printed` by the MAIN TRAINING script.):
    (2 GPUS for VLLM)
---
    CUDA_VISIBLE_DEVICES='4,5' \
    REMOTE_VLLM_INIT_MODEL='/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-7B-Instruct/' \
    REMOTE_VLLM_NCCL_LINK=True \
    REMOTE_VLLM_GPUS=2 \
    REMOTE_VLLM_GPU_FRAG=0.9 \
    REMOTE_VLLM_MAX_MODEL_LEN=4096 \
    REMOTE_VLLM_MAX_LORA_RANK=0 \   # <--- never change this, even if you use lora
    REMOTE_VLLM_TEMPERATURE=0.9 REMOTE_VLLM_NUM_GENERATION=8 \
    python3 /your/path/to/trl/extras/remote_vllm_helper.py






------------------------------------------------------------------------------------------------------------
2 machine       | 1 for training, 1 for VLLM      | using NCCL to deliver param updates
------------------------------------------------------------------------------------------------------------

---
(1) start MAIN TRAINING script:
    (on machine 1, all 8 gpus for training)
---
    CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' \
    accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
        --num_processes=8 \
        grpo_with_remote_vllm.py \
        --model_name_or_path /mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-7B-Instruct/ \
        --dataset_name "trl-internal-testing/zen" \
        --output_dir './mytests' \
        --bf16 \
        --use_remote_vllm=True \
        --vllm_max_model_len 4096 \
        --remote_vllm_num_gpus=1 \
        --remote_vllm_ip_port='22.6.225.225:8000'

---
(2) start VLLM script (do not run the commandline below, it's only a demo, the true commandline will be `printed` by the MAIN TRAINING script.):
    (on machine 2, 1 GPU for VLLM)
---
    >> the commandline will be `printed` by the MAIN TRAINING script.



------------------------------------------------------------------------------------------------------------
3+ machine       | 2+ for training, 1 for VLLM      | using NCCL to deliver param updates
------------------------------------------------------------------------------------------------------------

---
(1) start MAIN TRAINING script:
    (on machine 1~(n-1), all gpus for training)
---
    CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' \
    accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
        --machine-rank="x" \
        --main_process_ip="x.x.x.x" \
        --num_processes=16 \
        grpo_with_remote_vllm.py \
        --model_name_or_path /mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-7B-Instruct/ \
        --dataset_name "trl-internal-testing/zen" \
        --output_dir './mytests' \
        --bf16 \
        --use_remote_vllm=True \
        --vllm_max_model_len 4096 \
        --remote_vllm_num_gpus=1 \
        --remote_vllm_ip_port='x.x.x.x:8000'
---
(3) start VLLM script:
    (on machine n)
    >> the commandline will be `printed` by the MAIN TRAINING script.
---

"""


import argparse
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

def len_reward(completions, **kwargs):
    print(completions)
    return [len(completion) for completion in completions]


def main(script_args, training_args, model_args):
    # Load the dataset
    # dataset = load_dataset("/root/.cache/huggingface/hub/datasets--trl-internal-testing--zen")
    dataset = load_dataset("/root/.cache/huggingface/hub/datasets--trl-internal-testing--zen/snapshots/47aee340f33dd6161e2baa618240b8514666c822/standard_prompt_only")


    # Load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )


    # Initialize the GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[len_reward],
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, GRPOConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("grpo", help="Run the GRPO training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
