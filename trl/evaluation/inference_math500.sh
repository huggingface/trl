mkdir -p output

model_name="DeepSeek-R1-Distill-Qwen-1.5B-Open-Math-On-Policy-Distillation"

mkdir -p output/$model_name

# IFEval
python ./generate_api_answers/infer_multithread.py --input_file "./data/math500.jsonl" --output_file "./output/$model_name/math500_bz1.jsonl"  --base_url "http://127.0.0.1:8030/v1" --model_name $model_name --n_samples 1