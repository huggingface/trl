# RLHF pipeline for the creation of "stack llama" a stack exchange llama-7b model.
There were three main steps to the training process:
1. Supervised fine-tuning of the base llama-7b model to create llama-7b-se - sft_stack_exchange_peft.py
2. Reward modeling using dialog pairs from the SE dataset using the llama-7b-se to create llama-7b-se-rm - reward_modeling_peft.py
3. RL fine-tuning of llama-7b-se with the llama-7b-se-rm reward model - rl_finetuning_peft.py

LoRA layers were using at all stages to reduce memory requirements. 
At each stage the peft adapter layers were merged with the base model, using: `merge_peft_adapter.py --model_id=XXX`

