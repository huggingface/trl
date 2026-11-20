# W1-M recommended arm: w1-S-recommended.sh (REINFORCE++-baseline, k2 KL 1e-5 with a colocated reference, dynamic
# filtering, partial rollout) with the same M overrides as w1-M.sh. The reference model is a full bf16 replica per
# training GPU under ZeRO-2 (OpenRLHF only shards it under ZeRO-3), ~19 GB, on top of the actor.
# Launch: SIZE=M CONFIG_NAME=w1-M-recommended TAG=recommended STOP_AT_STEP=200 b200-queue ... launch/openrlhf/w1.sbatch
source "$BENCH/configs/openrlhf/w1-S-recommended.sh"
W1_ARGS+=(
   --ds.zero_stage 2
   --ckpt.save_steps "${SAVE_STEPS:-50}"
)
