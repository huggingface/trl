if [[ -z $CUDA_VISIBLE_DEVICES ]]; then
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    gpu_count=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi
echo "Available GPU count: $gpu_count"

mkdir -p output

model_name="jetlm-2B_instruct_hf_2e-6-packing"
model_name_or_path="/homes/yechengw/workspace/jet-pretrain/checkpoints/jetlm/2B_instruct_hf_2e-6-packing"


mkdir -p output/$model_name

CUDA_VISIBLE_DEVICES=0 torchrun --nnodes 1 --nproc_per_node=1 --rdzv_endpoint localhost:29800 \
 ./generate_api_answers/meta_infer_multithread.py --input_file "./data/math500.jsonl" --output_file "./output/$model_name/math500_bz1.jsonl" --model_name_or_path $model_name_or_path \
 --n_samples 1 --max_tokens 16384