# NeMo Gym TRL GRPO integration 

Multi-step GRPO with TRL and NeMo Gym.

## Setup

1. Launch vLLM server:
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 trl vllm-serve \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --trust-remote-code
```

2. Start NeMo Gym servers
```
ng_run "+config_paths=[resources_servers/workplace_assistant/configs/workplace_assistant.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```


3. Run training:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config.yaml
```

can do dp=7 with 7/8 gpus for vllm server. Havent gotten multigpu training backend to work despite docs says it works https://huggingface.co/docs/trl/main/en/vllm_integration#modes-of-using-vllm-during-training