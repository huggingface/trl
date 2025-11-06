FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip uv
RUN uv pip install --system --no-cache "git+https://github.com/huggingface/trl.git#egg=trl[liger,peft,vlm]"
RUN uv pip install --system hf_transfer kernels liger_kernel peft trackio