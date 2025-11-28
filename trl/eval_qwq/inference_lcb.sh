mkdir -p output

model_name="Qwen2.5-1.5B-Instruct"

mkdir -p output/$model_name

# livebench 2408-2502 (repeated sample 8 times)
python ./generate_api_answers/infer_multithread.py --input_file "./data/livecodebench_v5.jsonl" --output_file "./output/$model_name/livecodebench_v5_bz8.jsonl"  --base_url "http://127.0.0.1:8030/v1" --model_name $model_name --n_samples 8