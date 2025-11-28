PROMPT_TYPE="cot"
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-1.5B"
OUTPUT_DIR=results/eval/${MODEL_NAME_OR_PATH}

SPLIT="test"
NUM_TEST_SAMPLE=-1

# English open datasets
DATA_NAME="eleuther_math"
TOKENIZERS_PARALLELISM=false \
python3 -u trl/evaluation/math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite \
    --num_shots 5 \
    --evaluate_max_workers 32