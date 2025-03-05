#!/bin/bash
# This script runs SPIN with pre-generated data. To generate your own data, see https://github.com/uclaml/SPIN/blob/main/spin/generate_vllm.py

OUTPUT_DIR="test_spin/"
MAX_STEPS=500
BATCH_SIZE=1
SEQ_LEN=128

export WANDB_DISABLED=true

# Handle extra arguments in case one passes accelerate configs.
EXTRA_ACCELERATE_ARGS=""
EXTRA_TRAINING_ARGS="--use_peft"

# This is a hack to get the number of available GPUs
NUM_GPUS=4

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

iter_num=3
for i in $(seq 0 $iter_num); do
    echo "Running Iter ${i}"
    if [ "$i" -eq 0 ]; then
        MODEL_NAME="alignment-handbook/zephyr-7b-sft-full"
    else
        MODEL_NAME="UCLA-AGI/zephyr-7b-sft-full-SPIN-iter$((i-1))"
    fi
    DATASET_NAME="KaixuanJi/SPIN_iter${i}"

    CMD="""
    accelerate launch $EXTRA_ACCELERATE_ARGS \
        --num_processes $NUM_GPUS \
        --mixed_precision 'fp16' \
        `pwd`/trl/scripts/dpo.py \
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
    done

exit 0
