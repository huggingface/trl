# W1-S "recommended" arm (SHARED_W1 rev. 1): the shared arm (w1-S.sh) with loss, advantage, async and zero-advantage
# handling switched to what OpenRLHF itself recommends for async RLVR: README "REINFORCE++-baseline (best for RLVR)" and
# the flags of examples/scripts/train_reinforce_baseline_ray_agent_async.sh (its async RLVR recipe). argparse keeps the
# last value of a repeated flag, so these are appended after the shared ones. Differences are listed in EQUIVALENCE.md.
# Launch: CONFIG_NAME=w1-S-recommended TAG=recommended STOP_AT_STEP=200 b200-queue ... launch/openrlhf/w1.sbatch
source "$BENCH/configs/openrlhf/w1-S.sh"
W1_ARGS+=(
   # advantage: REINFORCE++-baseline = group-mean subtraction, then whitening over the whole batch (mean/std of tokens)
   --algo.advantage.estimator reinforce_baseline
   # clip: CLI default symmetric eps 0.2 (the async recipe sets none)
   --actor.eps_clip_low_high 0.2 0.2
   # KL: k2 KL as a loss against a frozen reference, coef 1e-5, as in the async recipe; the reference model shares the
   # 4 training GPUs
   --algo.kl.init_coef 1e-5
   --algo.kl.use_loss
   --algo.kl.estimator k2
   --ref.num_nodes 1
   --ref.num_gpus_per_node "$NGPU_TRAIN"
   --train.colocate_actor_ref
   # zero-advantage groups: DAPO dynamic filtering, groups whose mean reward is 0 or 1 are dropped and replaced by new
   # prompts until 32 groups pass
   --algo.dynamic_filtering_enable
   --algo.dynamic_filtering_range 0.0 1.0
   # filtering consumes extra prompts: give it the whole pool (w1.sbatch stops the run at STOP_AT_STEP=200)
   --data.max_samples 100000000
   # async: partial rollout (vLLM pause/resume for the weight sync instead of the lock) + token IS mask, as in the recipe
   --train.partial_rollout_enable
)
