# model_path="Qwen/Qwen2.5-1.5B" # or path to your local checkpoint
# model_name="qwen2.5-1.5B"

model_path="Qwen/Qwen2.5-Math-1.5B" # or path to your local checkpoint
model_name="qwen2.5-math-1.5B"

export VLLM_WORKER_MULTIPROC_METHOD=spawn

python -m vllm.entrypoints.openai.api_server \
    --model $model_path \
    --trust-remote-code \
    --served-model-name $model_name \
    --port 8030 \
    --data-parallel-size 4