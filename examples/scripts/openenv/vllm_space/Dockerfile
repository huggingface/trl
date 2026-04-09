FROM vllm/vllm-openai:latest

# Install git (needed to pip install from git repos), then TRL from delta-weight-sync branch
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install "trl @ git+https://github.com/huggingface/trl.git@delta-weight-sync"
RUN pip install "transformers==5.2.0"

# HF Spaces expects port 7860
EXPOSE 7860

# HF Spaces runs as uid 1000 without a passwd entry — PyTorch needs USER set
ENV VLLM_SERVER_DEV_MODE=1
ENV USER=user
ENV HOME=/tmp
ENV HF_HOME=/tmp/hf_cache
ENV TORCH_HOME=/tmp/torch_cache
ENV XDG_CACHE_HOME=/tmp/.cache
ENV FLASHINFER_WORKSPACE_DIR=/tmp/flashinfer

ENTRYPOINT ["vllm", "serve", "Qwen/Qwen3-1.7B", \
    "--host", "0.0.0.0", \
    "--port", "7860", \
    "--worker-extension-cls", "trl.experimental.async_grpo.delta_engine.DeltaWorkerExtension", \
    "--weight-transfer-config", "{\"backend\":\"nccl\"}", \
    "--max-model-len", "32768", \
    "--enforce-eager", \
    "--gpu-memory-utilization", "0.8", \
    "--logprobs-mode", "processed_logprobs"]
