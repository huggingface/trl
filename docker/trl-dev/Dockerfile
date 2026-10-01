FROM vllm/vllm-openai:v0.30.0
ENTRYPOINT []
RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip uv
RUN uv pip install --system --no-cache "git+https://github.com/huggingface/trl.git#egg=trl[liger,peft,vlm]"
RUN uv pip install --system kernels liger_kernel peft trackio
