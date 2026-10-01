FROM nvidia/cuda:13.0.3-devel-ubuntu24.04
RUN apt-get update && apt-get install -y git python3-dev python3-pip python-is-python3 && rm -rf /var/lib/apt/lists/*
RUN pip install --break-system-packages uv
RUN uv pip install --system --break-system-packages --no-cache "git+https://github.com/huggingface/trl.git#egg=trl[liger,peft,vlm,vllm]"
RUN uv pip install --system --break-system-packages kernels liger_kernel peft trackio
