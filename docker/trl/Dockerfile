FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip uv
RUN uv pip install --system trl[liger,peft,vlm] kernels trackio