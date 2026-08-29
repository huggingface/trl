FROM vllm/vllm-openai:v0.26.0
ENTRYPOINT []
WORKDIR /workspace
RUN apt-get update && apt-get install -y git build-essential python-is-python3 && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip uv
RUN uv pip install --system trl[liger,peft,vlm] kernels trackio