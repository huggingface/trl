# VIMPO Trainer

[VIMPO: Value-Implicit Policy Optimization for LLMs](https://huggingface.co/papers/2606.20008) is a critic-free RLVR method that keeps GRPO-style grouped rollouts while recovering token-level credit assignment from a policy-implied value recurrence. VIMPO trains a terminal value loss against the centered final reward and can add a PPO-style actor branch using detached policy-implied advantages.

To use VIMPO, you can use the [`VIMPOTrainer`] class in `trl.experimental.vimpo`.

## Usage

```python
from trl.experimental.vimpo import VIMPOConfig, VIMPOTrainer

training_args = VIMPOConfig(
    vimpo_beta=5e-4,
    vimpo_actor_coeff=5e-3,
    num_generations=8,
)
trainer = VIMPOTrainer(
    model="Qwen/Qwen3-0.6B",
    reward_funcs=...,
    train_dataset=...,
    args=training_args,
)
trainer.train()
```

VIMPO reuses GRPO's generation and reward-function interface. The implementation follows the paper's raw VIMPO setting: exact `KL(pi_theta || pi_ref)`, detached KL in the terminal value loss, a frozen reference policy, centered rewards without reward scaling, and a PPO actor branch controlled by `vimpo_actor_coeff` with normalized detached actor advantages.

## Expected dataset type

VIMPO requires a [prompt-only dataset](dataset_formats#prompt-only). The [`experimental.vimpo.VIMPOTrainer`] supports both [conversational](dataset_formats#conversational) and [standard](dataset_formats#standard) dataset formats. When provided with a conversational dataset, the trainer will automatically apply the chat template to the dataset.

## Example script

Use [`examples/scripts/vimpo.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/vimpo.py) to launch VIMPO training on [AI-MO/NuminaMath-TIR](https://huggingface.co/datasets/AI-MO/NuminaMath-TIR) with verifiable math rewards.

```bash
accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/vimpo.py \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --output_dir vimpo-Qwen3-0.6B \
    --learning_rate 1e-5 \
    --dtype bfloat16 \
    --max_completion_length 1024 \
    --use_peft \
    --lora_target_modules "q_proj", "v_proj" \
    --per_device_train_batch_size 8 \
    --num_generations 8 \
    --vimpo_beta 5e-4 \
    --vimpo_actor_coeff 5e-3
```

## VIMPOTrainer

[[autodoc]] experimental.vimpo.VIMPOTrainer
    - train
    - save_model
    - push_to_hub

## VIMPOConfig

[[autodoc]] experimental.vimpo.VIMPOConfig
