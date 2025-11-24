model_path="Qwen/Qwen2.5-1.5B" # or path to your local checkpoint
model_name="qwen2.5-1.5B"
num_gpus=4

export VLLM_WORKER_MULTIPROC_METHOD=spawn

python -m vllm.entrypoints.openai.api_server \
    --model $model_path \
    --trust-remote-code \
    --served-model-name $model_name \
    --tensor-parallel-size $num_gpus \
    --enforce-eager \
    --port 8030
