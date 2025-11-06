FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
RUN pip install --upgrade pip uv
RUN uv pip install --system trl[liger,peft,vlm] hf_transfer kernels trackio