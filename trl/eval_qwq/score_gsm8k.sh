model_name="Qwen/Qwen3-4B"

res_dir=eval_res/$model_name

mkdir -p $res_dir

python  ./eval/eval.py --input_path ./output/$model_name/GSM8k-Qwen3-4B.jsonl --cache_path $res_dir/GSM8k-Qwen3-4B.jsonl  --task_name "math_opensource/gsm8k" > $res_dir/GSM8k-Qwen3-4B_res_result.txt
