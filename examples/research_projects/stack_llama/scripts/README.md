# RLHF pipeline for the creation of StackLLaMa: a Stack exchange llama-7b model.
There were three main steps to the training process:
1. Supervised fine-tuning of the base llama-7b model to create llama-7b-se:
    - `torchrun --nnodes 1  --nproc_per_node 8 examples/stack_llama/scripts/supervised_finetuning.py --model_path=<LLAMA_MODEL_PATH> --streaming --no_gradient_checkpointing --learning_rate 1e-5 --max_steps 5000 --output_dir ./llama-se`
2. Reward modeling using dialog pairs from the SE dataset using the llama-7b-se to create llama-7b-se-rm:
    - `torchrun --nnodes 1  --nproc_per_node 8 examples/stack_llama/scripts/reward_modeling.py --model_name=<LLAMA_SE_MODEL>`
3. RL fine-tuning of llama-7b-se with the llama-7b-se-rm reward model:
    - `accelerate launch --multi_gpu --num_machines 1  --num_processes 8 examples/stack_llama/scripts/rl_training.py --log_with=wandb --model_name=<LLAMA_SE_MODEL> --reward_model_name=<LLAMA_SE_RM_MODEL> --adafactor=False --tokenizer_name=<LLAMA_TOKENIZER> --save_freq=100 --output_max_length=128 --batch_size=8 --gradient_accumulation_steps=8 --batched_gen=True --ppo_epochs=4 --seed=0 --learning_rate=1.4e-5 --early_stopping=True --output_dir=llama-se-rl-finetune-128-8-8-1.4e-5_adam`


LoRA layers were using at all stages to reduce memory requirements. 
At each stage the peft adapter layers were merged with the base model, using: 
```shell
python examples/stack_llama/scripts/merge_peft_adapter.py --adapter_model_name=XXX --base_model_name=YYY --output_name=ZZZ
```
Note that this script requires `peft>=0.3.0`.

For access to the base llama-7b model, please see Meta's [release](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) and [request form](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform).
