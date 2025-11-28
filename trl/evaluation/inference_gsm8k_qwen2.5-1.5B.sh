model_name="qwen2.5-1.5B"

mkdir -p results/eval/$model_name

python trl/evaluation/generate_api_answers/evaluate_gsm8k_multinode.py \
 --output_file "results/eval/$model_name/gsm8k_fix5.jsonl" \
 --few_shot_prompt_file trl/evaluation/generate_api_answers/gsm8k_prompt.txt \
 --base_url "http://127.0.0.1:8030/v1" \
 --model_name $model_name \
 --max_workers 32