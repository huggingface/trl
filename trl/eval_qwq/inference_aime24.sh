mkdir -p output

model_name="DeepSeek-R1-Distill-Qwen-1.5B"

mkdir -p output/$model_name

# aime24 (repeated sample 64 times)
python ./generate_api_answers/infer_multithread.py --input_file "./data/aime24.jsonl" --output_file "./output/$model_name/aime24_bz64.jsonl"  --base_url "http://127.0.0.1:8030/v1" --model_name $model_name --n_samples 1