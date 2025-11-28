if [[ -z $CUDA_VISIBLE_DEVICES ]]; then
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    gpu_count=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi
echo "Available GPU count: $gpu_count"

mkdir -p output

model_name="Qwen/Qwen3-1.7B-On-Policy-Distillation-Debug"
model_path="/homes/yechengw/workspace/open-r1/data/Qwen3-1.7B-Open-Math-On-Policy-Distillation"

mkdir -p output/$model_name

torchrun --nnodes 1 --nproc_per_node=$gpu_count --rdzv_endpoint localhost:29800 \
 ./generate_api_answers/meta_infer_multithread.py --input_file "./data/gsm8k.jsonl" --output_file "./output/$model_name/gsm8k_bz1.jsonl" --model_name_or_path $model_path \
 --n_samples 1 --max_tokens 2048