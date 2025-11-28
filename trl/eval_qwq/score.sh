model_name="jetlm-2B-instruct-hf-5e-6-instruct"

res_dir=eval_res/$model_name

mkdir -p $res_dir

# python  ./eval/eval.py --input_path ./output/$model_name/aime24_bz64.jsonl --cache_path $res_dir/aime24_bz64.jsonl  --task_name "math_opensource/aime24" > $res_dir/aime24_bz64_res_result.txt

# python  ./eval/eval.py --input_path ./output/$model_name/aime25_bz64.jsonl --cache_path $res_dir/aime25_bz64.jsonl  --task_name "math_opensource/aime25" > $res_dir/aime25_bz64_res_result.txt

# # download all test cases
# python ./data/process_data.py
# # Note: running all code test cases can be very slow (more than 4 hours)
# python  ./eval/eval.py --input_path ./output/$model_name/livecodebench_v5_bz8.jsonl --cache_path $res_dir/livecodebench_v5_bz8.jsonl  --task_name "livecodebench" > $res_dir/livecodebench_v5_bz8_res_result.txt

python  ./eval/eval.py --input_path ./output/$model_name/ifeval_bz1.jsonl --cache_path $res_dir/ifeval_bz1.jsonl  --task_name "ifeval" > $res_dir/ifeval_bz1_res_result.txt
