FROM python:3.12
ARG VERSION
RUN pip install --upgrade pip uv
RUN uv pip install --system trl[liger,peft,vlm,vllm]==${VERSION} kernels trackio
