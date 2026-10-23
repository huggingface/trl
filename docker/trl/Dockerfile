FROM nvidia/cuda:13.0.3-devel-ubuntu24.04
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv
ARG VERSION
ENV VIRTUAL_ENV=/opt/venv UV_PYTHON_INSTALL_DIR=/opt/uv/python PATH=/opt/venv/bin:$PATH
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN uv venv --python 3.13 --seed $VIRTUAL_ENV
# Without this floor, uv may pick vLLM 0.30.0 (no transformers cap) with the latest transformers (not tested with vLLM 0.30.0): https://github.com/huggingface/trl/issues/7819
RUN uv pip install --no-cache trl[kernels,liger,peft,vlm,vllm]==${VERSION} "vllm>=0.31.0" trackio
