PROMPT_TYPE="qwen25-math-cot"
MODEL_TAG="sft/openr1-math-220k/qwen2.5-1.5B/checkpoint-538"
MODEL_NAME_OR_PATH="results/${MODEL_TAG}"
OUTPUT_DIR=results/eval/${MODEL_TAG}

SPLIT="test"
NUM_TEST_SAMPLE=-1

# English open datasets
DATA_NAME="math500"
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
    --evaluate_max_workers 32