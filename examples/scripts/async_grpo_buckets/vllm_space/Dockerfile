# vLLM inference Space for async GRPO with bucket weight sync.
#
# Serves Qwen/Qwen3-1.7B and applies sparse weight patches in place from an HF Storage Bucket via the
# `hf_bucket` weight-transfer backend (the HFBucketWorkerExtension registers it in the worker process).
#
# Requirements baked in here:
#   * a vLLM with the sparse weight-transfer API (vllm-project/vllm#40096) — installed from the nightly index,
#     since it is not in a stable release yet.
#   * `--model-impl transformers`  -> vLLM's runtime param names match the trainer's HF names, so every param is
#     addressable by the in-place sparse apply (no fuse/unfuse remap).
#   * `VLLM_USE_V2_MODEL_RUNNER=0`  -> the in-place sparse apply (apply_sparse_weight_patches) is V1-runner only.
FROM vllm/vllm-openai:latest

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# vLLM 0.23.0 ships sparse weight transfer (vllm-project/vllm#40096), then TRL (v3 branch) + a recent transformers.
RUN pip install "vllm>=0.23.0"
RUN pip install "transformers>=5.2.0"
RUN pip install "trl @ git+https://github.com/huggingface/trl.git@delta-weight-sync-v3"

# HF Spaces serves on port 7860 and runs as uid 1000 without a passwd entry — set HOME/USER + writable caches.
EXPOSE 7860
ENV VLLM_SERVER_DEV_MODE=1
ENV VLLM_USE_V2_MODEL_RUNNER=0
ENV USER=user
ENV HOME=/tmp
ENV HF_HOME=/tmp/hf_cache
ENV TORCH_HOME=/tmp/torch_cache
ENV XDG_CACHE_HOME=/tmp/.cache
ENV FLASHINFER_WORKSPACE_DIR=/tmp/flashinfer

ENTRYPOINT ["vllm", "serve", "Qwen/Qwen3-1.7B", \
    "--host", "0.0.0.0", \
    "--port", "7860", \
    "--model-impl", "transformers", \
    "--worker-extension-cls", "trl.experimental.async_grpo.delta_engine.HFBucketWorkerExtension", \
    "--weight-transfer-config", "{\"backend\":\"hf_bucket\"}", \
    "--max-model-len", "8192", \
    "--gpu-memory-utilization", "0.85", \
    "--logprobs-mode", "processed_logprobs"]
