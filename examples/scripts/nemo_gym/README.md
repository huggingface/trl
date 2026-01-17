# Post-training with NeMo Gym and TRL

This integration supports training language models in NeMo-Gym environments using TRL GRPO.

## Interactive single node 

1. Launch vLLM server:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 trl vllm-serve \
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
CUDA_VISIBLE_DEVICES=4 python train.py --config config.yaml
```

## Multinode with slurm

See submit.sh for a multinode example!

```


```