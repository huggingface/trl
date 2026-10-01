FROM vllm/vllm-openai:v0.30.0
ENTRYPOINT []
ARG VERSION
RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip uv
RUN uv pip install --system trl[liger,peft,vlm]==${VERSION} kernels trackio
