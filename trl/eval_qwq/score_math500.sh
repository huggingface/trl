model_name="DeepSeek-R1-Distill-Qwen-1.5B-Open-Math-On-Policy-Distillation"

res_dir=eval_res/$model_name

mkdir -p $res_dir

python  ./eval/eval.py --input_path ./output/$model_name/math500_bz1.jsonl --cache_path $res_dir/math500_bz1.jsonl  --task_name "math_opensource/math500" > $res_dir/math500_bz1_res_result.txt

cat $res_dir/math500_bz1_res_result.txt
