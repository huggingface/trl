if [[ -z $CUDA_VISIBLE_DEVICES ]]; then
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    gpu_count=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi
echo "Available GPU count: $gpu_count"

mkdir -p output

model_name="jet-ai/Jet-Nemotron-2B"

mkdir -p output/jetlm-2B-hf

# IFEval
CUDA_VISIBLE_DEVICES=0  torchrun --nnodes 1 --nproc_per_node=1 --rdzv_endpoint localhost:29700 \
 ./generate_api_answers/meta_infer_multithread.py --input_file "./data/math.jsonl" --output_file "./output/jetlm-2B-hf/math500_bz1.jsonl" --model_name_or_path $model_name \
 --n_samples 1 --max_tokens 2048