pip install uv
uv venv .venv
source .venv/bin/activate
git clone https://github.com/huggingface/trl/
cd trl
git clone https://github.com/huggingface/OpenEnv.git OpenEnv
git -C OpenEnv checkout --detach 86a180ede21e044f7929b9a7783ad83aa67d83a3
uv pip install -e . trackio datasets ./OpenEnv
export PYTHONPATH="$PWD/OpenEnv/envs${PYTHONPATH:+:$PYTHONPATH}"
git remote add younes https://github.com/younesbelkada/trl.git
uv pip install vllm
uv pip install fastmcp