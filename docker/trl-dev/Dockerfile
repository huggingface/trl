FROM python:3.12
RUN pip install --upgrade pip uv
RUN uv pip install --system --no-cache "git+https://github.com/huggingface/trl.git#egg=trl[liger,peft,vlm,vllm]"
RUN uv pip install --system kernels liger_kernel peft trackio
