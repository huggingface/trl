FROM nvidia/cuda:13.0.3-devel-ubuntu24.04
COPY --from=ghcr.io/astral-sh/uv:0.12.24 /uv /usr/local/bin/uv
ARG VERSION
ENV VIRTUAL_ENV=/opt/venv UV_PYTHON_INSTALL_DIR=/opt/uv/python PATH=/opt/venv/bin:$PATH
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN uv venv --python 3.13 --seed $VIRTUAL_ENV
RUN uv pip install --no-cache trl[liger,peft,vlm,vllm]==${VERSION} kernels trackio
