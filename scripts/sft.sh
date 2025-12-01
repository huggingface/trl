GROUP_NAME=openr1-math-220k/qwen2.5-1.5B_test3
JOB_TYPE=sft
RUN_NAME=${JOB_TYPE}/${GROUP_NAME}
OUTPUT_DIR=results/${RUN_NAME}


accelerate launch --config_file=trl/accelerate_configs/zero1.yaml --num_processes 4 trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B \
    --dataset_name dataset/llm_rl/OpenR1-Math-220k \
    --learning_rate 2.0e-5 \
    --num_train_epochs 5 \
    --packing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eos_token '<|im_end|>' \
    --eval_strategy no \
    --save_strategy epoch \
    --eval_steps 100 \
    --output_dir ${OUTPUT_DIR} \
    --attn_implementation flash_attention_2 \
    --dtype bfloat16 \
    --max_length 32768 \
    --assistant_only_loss true \
    --chat_template_path trl/chat_templates/qwen2.5_sft_assistant_only.jinja \
    --resume_from_checkpoint True \
    --report_to none \
    --wandb_run_name $RUN_NAME \
    --wandb_mode online \
    --wandb_job_type $JOB_TYPE \
    --wandb_group $GROUP_NAME \
    --logging_steps 1 \
    --logging_first_step true \