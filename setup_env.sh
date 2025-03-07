#!/bin/bash
# Setup script for trl-jin with SGLang support

# Create virtual environment
echo "Creating Python virtual environment at ~/.python/trl..."
python3 -m venv ~/.python/trl

# Activate virtual environment
echo "Activating virtual environment..."
source ~/.python/trl/bin/activate

# Install uv package installer
echo "Installing uv package installer..."
pip install uv

# Install the package in development mode
echo "Installing trl in development mode..."
pip install -e .[dev]

# Install SGLang with all dependencies
echo "Installing SGLang with dependencies..."
python3 -m uv pip install "sglang[all] @ git+https://github.com/sgl-project/sglang.git/@main#egg=sglang&subdirectory=python" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python

echo "Setup complete! The environment is ready for use."
echo "Activate the environment in new terminals with: source ~/.python/trl/bin/activate" 