# W1-M (math, Qwen3.5-9B) shared arm: the W1-S flags unchanged (w1-S.sh), with MODEL and DATA pointing at the 9B
# snapshot and its own pre-rendered prompts (w1.sbatch picks both from SIZE=M; the 9B chat template differs from the 2B
# one, so the M parquet is rendered from the 9B processor). argparse keeps the last value of a repeated flag, so the M
# overrides are appended. M-vs-S differences: EQUIVALENCE.md section "M".
# Launch: SIZE=M b200-queue --nodes=1 --gres=gpu:8 --exclusive launch/openrlhf/w1.sbatch
source "$BENCH/configs/openrlhf/w1-S.sh"
W1_ARGS+=(
   # ZeRO-2 stays: 9B bf16 params (19 GB) replicated + fp32 master/Adam sharded over 4 ranks (~28 GB/rank) + grads
   # leaves ~100 GB of the 180 GB for the 16k-token micro-batch (fp32 logits alone are ~16.5 GB)
   --ds.zero_stage 2
   # SAVE_STEPS only for smokes (time a 9B save); the run keeps 50
   --ckpt.save_steps "${SAVE_STEPS:-50}"
)
