mkdir -p output

model_name="Qwen2.5-Math-1.5B-Open-Math-On-Policy-Distillation"

mkdir -p output/$model_name

# aime24 (repeated sample 64 times)
python ./generate_api_answers/evaluate_gsm8k_multinode.py --output_file "./output/$model_name/gsm8k.jsonl"  --base_url "http://127.0.0.1:8030/v1" --model_name $model_name