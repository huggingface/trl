FROM nvidia/cuda:13.0.3-devel-ubuntu24.04
ARG VERSION
ENV PIP_BREAK_SYSTEM_PACKAGES=1 UV_BREAK_SYSTEM_PACKAGES=1
RUN apt-get update && apt-get install -y git python3-dev python3-pip python-is-python3 && rm -rf /var/lib/apt/lists/*
RUN pip install uv
RUN uv pip install --system trl[liger,peft,vlm,vllm]==${VERSION} kernels trackio
