model_name="jetlm-2B-instruct-hf-5e-6"

res_dir=eval_res/$model_name

mkdir -p $res_dir

python  ./eval/eval.py --input_path ./output/$model_name/math_bz1.jsonl --cache_path $res_dir/math_bz1.jsonl  --task_name "math_opensource/math" > $res_dir/math_bz1_res_result.txt
