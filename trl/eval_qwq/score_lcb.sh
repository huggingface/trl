model_name="Qwen2.5-1.5B-Instruct"

res_dir=eval_res/$model_name

mkdir -p $res_dir

# download all test cases
python ./data/process_data.py
# Note: running all code test cases can be very slow (more than 4 hours)
python  ./eval/eval.py --input_path ./output/$model_name/livecodebench_v5_bz8.jsonl --cache_path $res_dir/livecodebench_v5_bz8.jsonl  --task_name "livecodebench" > $res_dir/livecodebench_v5_bz8_res_result.txt
