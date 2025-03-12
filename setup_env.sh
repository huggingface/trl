#!/bin/bash
# Setup script for trl_GRPO with SGLang support
# Doc Link: https://docs.sglang.ai/start/install.html

# Create virtual environment
echo "Creating Python virtual environment at ~/.python/trl..."
python3 -m venv ~/.python/trl

# Activate virtual environment
echo "Activating virtual environment..."
source ~/.python/trl/bin/activate

# Install uv package installer
echo "Installing uv package installer..."
pip install uv

# Install PyTorch with CUDA 12.1 support
# System CUDA Version needs to be consistent with pip CUDA Version
echo "Installing PyTorch with CUDA 12.1 support..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install the package in development mode
echo "Installing trl in development mode..."
pip install -e .[dev]

# Install SGLang with all dependencies
echo "Installing SGLang with dependencies..."
python3 -m uv pip install "sglang[all] @ git+https://github.com/sgl-project/sglang.git/@main#egg=sglang&subdirectory=python" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python

echo "Setup complete! The environment is ready for use."
echo "Activate the environment in new terminals with: source ~/.python/trl/bin/activate" 