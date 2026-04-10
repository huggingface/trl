the goal is test the scaling limits for SFT in TRL?
Concretely, we could start with Qwen3 4B, 30B, 235B and see at what point one OOMs for SFT with:

- BS = 1
- FSDP2 vs DeepSpeed ZeRO-3
- Scaling context with CP (FSDP2) or Ulysses (DeepSpeed)
- Optional: FullFT vs LoRA

We will be using SFT trainer.

- The dataset is not so important but anything which allows you to test context at 32k tokens would be a good ballpark, e.g. a small subset of https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M

- we will be using Slurm.

- I want to output a table in this format :

```csv
Model,Context length (k tokens),Nodes,Distributed backend,CPU Offload,DP,TP,PP,CP,EP,MFU,TPS,TPS/GPU,wandb,Success?
Qwen3 4B Thinking 2507,16,2,fsdp2,false,16,1,1,1,1,7.13,52563.78,3285.24,on,Yes
Qwen3 4B Thinking 2507,16,2,fsdp2,false,8,1,1,2,1,18.63,68700.98,4293.81,on,Yes
Qwen3 4B Thinking 2507,16,2,fsdp2,false,4,1,1,4,1,29.77,54845.28,3427.83,on,Yes
Qwen3 4B Thinking 2507,32,2,fsdp2,false,8,1,1,2,1,20.79,46658.22,2916.76,on,Yes
Qwen3 4B Thinking 2507,32,2,fsdp2,false,4,1,1,4,1,37.16,41699.09,2606.19,on,Yes
Qwen3 4B Thinking 2507,32,4,fsdp2,false,8,4,1,1,1,4.92,44129.72,5516.22,on,Yes
Qwen3 4B Thinking 2507,32,4,fsdp2,false,4,8,1,1,1,3.68,33053.64,8263.41,on,Yes
Qwen3 4B Thinking 2507,32,4,fsdp2,false,16,1,1,2,1,20.28,91068.00,2845.88,on,Yes
Qwen3 4B Thinking 2507,32,4,fsdp2,false,8,1,1,4,1,35.96,80712.38,2522.26,on,Yes
Qwen3 4B Thinking 2507,32,4,fsdp2,false,4,1,1,8,1,56.39,63277.65,1977.43,on,Yes
Qwen3 30B-A3B Thinking 2507,16,4,fsdp2,false,8,1,1,1,4,8.76,133978.46,4186.83,on,Yes
Qwen3 30B-A3B Thinking 2507,16,4,fsdp2,false,4,1,1,1,8,7.50,114731.46,3585.36,on,Yes
Qwen3 30B-A3B Thinking 2507 backend match non-TE,16,4,fsdp2,false,8,1,1,1,4,0.81,14018.59,438.08,on,Yes
Qwen3 30B-A3B Thinking 2507 backend match TE,16,4,fsdp2,false,8,1,1,1,4,0.82,14138.05,441.81,on,Yes
Qwen3 30B-A3B Thinking 2507 backend match non-TE,16,4,fsdp2,false,4,1,1,1,8,0.67,11589.55,362.17,on,Yes
Qwen3 30B-A3B Thinking 2507 backend match TE,16,4,fsdp2,false,4,1,1,1,8,0.67,11617.69,363.05,on,Yes
Qwen3 30B-A3B Thinking 2507 TE speed FlashAdamW NVTE_FLASH_ATTN=0,16,4,fsdp2,false,8,1,1,1,4,0.89,15220.31,475.63,on,Yes
Qwen3 30B-A3B Thinking 2507 TE speed FlashAdamW NVTE_FLASH_ATTN=1,16,4,fsdp2,false,8,1,1,1,4,0.87,14824.62,463.27,on,Yes
Qwen3 30B-A3B Thinking 2507 TE speed FusedAdam NVTE_FLASH_ATTN=0,16,4,fsdp2,false,8,1,1,1,4,0.89,15200.82,475.03,on,Yes
Qwen3 30B-A3B Thinking 2507 TE speed FusedAdam NVTE_FLASH_ATTN=1,16,4,fsdp2,false,8,1,1,1,4,0.89,15157.92,473.69,on,Yes
Qwen3 30B-A3B Thinking 2507 TE speed FlashAdamW NVTE_FLASH_ATTN=0,16,4,fsdp2,false,4,1,1,1,8,0.77,11742.37,366.95,on,Yes
Qwen3 30B-A3B Thinking 2507 TE speed FlashAdamW NVTE_FLASH_ATTN=1,16,4,fsdp2,false,4,1,1,1,8,0.76,11645.99,363.94,on,Yes
Qwen3 30B-A3B Thinking 2507 TE speed FusedAdam NVTE_FLASH_ATTN=0,16,4,fsdp2,false,4,1,1,1,8,0.77,11731.61,366.61,on,Yes
Qwen3 30B-A3B Thinking 2507 TE speed FusedAdam NVTE_FLASH_ATTN=1,16,4,fsdp2,false,4,1,1,1,8,0.76,11638.11,363.69,on,Yes
Qwen3 30B-A3B Thinking 2507 TE,16,4,fsdp2,false,4,1,1,2,4,1.32,10100.15,315.63,on,Yes
Qwen3 30B-A3B Thinking 2507 TE,16,4,fsdp2,false,2,1,1,2,8,0.89,7558.68,236.21,on,Yes
Qwen3 30B-A3B Thinking 2507 TE,32,4,fsdp2,false,2,1,1,2,8,-,-,-,on,No (DeepEP timeout; corrected rerun failed in NCCL init)
Qwen3-235B-A22B Thinking 2507,16,32,fsdp2,false,2,1,4,1,32,-,-,-,on,No (cache and batch setup fixed; later failed with PP-group NCCL watchdog hang)
Qwen3-235B-A22B Thinking 2507 SFT non-TE,1,4,fsdp2,false,1,1,4,1,8,2.06,9723.56,1215.45,off,Yes
Qwen3-235B-A22B Thinking 2507 SFT non-TE,2,4,fsdp2,false,1,1,4,1,8,2.67,12202.56,1525.32,off,Yes
Qwen3-235B-A22B Thinking 2507 SFT non-TE,4,4,fsdp2,false,1,1,4,1,8,2.69,11374.48,1421.81,off,Yes
Qwen3-235B-A22B Thinking 2507 SFT non-TE,8,4,fsdp2,false,1,1,4,1,8,-,-,-,off,No (OOM on 4-node PP4/EP8 shape)
Qwen3-235B-A22B Thinking 2507 SFT non-TE,8,8,fsdp2,false,1,1,8,1,8,4.41,32476.96,4059.62,off,Yes
Qwen3-235B-A22B Thinking 2507 SFT non-TE,16,8,fsdp2,false,1,1,8,1,8,-,-,-,off,No (OOM in PP output merge)
Qwen3-235B-A22B Thinking 2507 SFT non-TE,16,8,fsdp2,false,1,1,4,1,16,-,-,-,off,No (EP peer-memory error over NVLink)
```

- First I want to compute the MFU of models in trl. I should have this in utils. Should implement something like : https://github.com/PrimeIntellect-ai/prime-rl/blob/main/src/prime_rl/trainer/perf.py to get the MFU. It should be a simple function. We can also precompute the needed FLOPS for each model size before

- We should use templates to start the jobs like : https://github.com/PrimeIntellect-ai/prime-rl/tree/main/src/prime_rl/templates
- We could be using configs like : https://github.com/huggingface/trl-jobs/tree/main/trl_jobs/configs for SFT training
- All scripts/ templates outside changes to trl should live in benchmark/
