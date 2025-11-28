model_name="DeepSeek-R1-Distill-Qwen-1.5B"

res_dir=eval_res/$model_name

mkdir -p $res_dir

python  ./eval/eval.py --input_path ./output/$model_name/aime24_bz64.jsonl --cache_path $res_dir/aime24_bz64.jsonl  --task_name "math_opensource/aime24" > $res_dir/aime24_bz64_res_result.txt