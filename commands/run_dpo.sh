#!/bin/bash
# This script runs an SFT example end-to-end on a tiny model using different possible configurations
# but defaults to QLoRA + PEFT
OUTPUT_DIR="test_dpo/"
MODEL_NAME="trl-internal-testing/tiny-random-LlamaForCausalLM"
DATASET_NAME="trl-internal-testing/hh-rlhf-helpful-base-trl-style"
MAX_STEPS=5
BATCH_SIZE=2
SEQ_LEN=128

# Handle extra arguments in case one passes accelerate configs.
EXTRA_ACCELERATE_ARGS=""
EXTRA_TRAINING_ARGS="""--use_peft \
    --load_in_4bit
"""

# This is a hack to get the number of available GPUs
NUM_GPUS=2

if [[ "${TRL_ACCELERATE_CONFIG}" == "" ]]; then
  EXTRA_ACCELERATE_ARGS=""
else
  EXTRA_ACCELERATE_ARGS="--config_file $TRL_ACCELERATE_CONFIG"
  # For DeepSpeed configs we need to set the `--fp16` flag to comply with our configs exposed
  # on `examples/accelerate_configs` and our runners do not support bf16 mixed precision training.
  if [[ $TRL_ACCELERATE_CONFIG == *"deepspeed"* ]]; then
    EXTRA_TRAINING_ARGS="--fp16"
  else
    echo "Keeping QLoRA + PEFT"
  fi
fi


CMD="""
accelerate launch $EXTRA_ACCELERATE_ARGS \
    --num_processes $NUM_GPUS \
    --mixed_precision 'fp16' \
    `pwd`/examples/scripts/dpo.py \
    --model_name_or_path $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --output_dir $OUTPUT_DIR \
    --max_steps $MAX_STEPS \
    --per_device_train_batch_size $BATCH_SIZE \
    --max_length $SEQ_LEN \
    $EXTRA_TRAINING_ARGS
"""

echo "Starting program..."

{ # try
    echo $CMD
    eval "$CMD"
} || { # catch
    # save log for exception 
    echo "Operation Failed!"
    exit 1
}
exit 0
