mkdir -p output

model_name="Qwen3-3B"

mkdir -p output/$model_name

# # aime24 (repeated sample 64 times)
# python ./generate_api_answers/infer_multithread.py --input_file "./data/aime24.jsonl" --output_file "./output/$model_name/aime24_bz64.jsonl"  --base_url "http://127.0.0.1:8030/v1" --model_name $model_name

# # aime25 (repeated sample 64 times)
# python ./generate_api_answers/infer_multithread.py --input_file "./data/aime25.jsonl" --output_file "./output/$model_name/aime25_bz64.jsonl"  --base_url "http://127.0.0.1:8030/v1" --model_name $model_name

# # livebench 2408-2502 (repeated sample 8 times)
# python ./generate_api_answers/infer_multithread.py --input_file "./data/livecodebench_v5.jsonl" --output_file "./output/$model_name/livecodebench_v5_bz8.jsonl"  --base_url "http://127.0.0.1:8030/v1" --model_name $model_name --n_samples 8

# IFEval
python ./generate_api_answers/infer_multithread.py --input_file "./data/ifeval.jsonl" --output_file "./output/$model_name/ifeval_bz1.jsonl"  --base_url "http://127.0.0.1:8030/v1" --model_name $model_name --n_samples 1