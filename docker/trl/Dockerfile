FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
RUN pip install --upgrade pip uv
RUN uv pip install --system trl[liger,peft,vlm] hf_transfer trackio
RUN uv pip install --system https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl