# W2-M (agentic TermGrade, Qwen3.5-9B) as `openrlhf.cli.train_ppo_ray` flags. Sourced by launch/openrlhf/w2.sbatch, which
# sets MODEL, DATA, RUN_DIR. Starts from the W1-S flags (w1-S.sh, same loss/optimizer/placement) and appends the W2
# overrides (argparse keeps the last value). Mapping and deviations: EQUIVALENCE.md "W2", envs/openrlhf/NOTES.md "W2-M".
NGPU_INF=4; NGPU_TRAIN=4
source "$BENCH/configs/openrlhf/w1-S.sh"
# drop the W1 reward file: W2's reward is the verifier score, returned by the agent executor
_A=(); _skip=0
for a in "${W1_ARGS[@]}"; do
   [ $_skip = 1 ] && { _skip=0; continue; }
   [ "$a" = --reward.remote_url ] && { _skip=1; continue; }
   _A+=("$a")
done
W1_ARGS=("${_A[@]}")
W1_ARGS+=(
   # multi-turn: OpenRLHF's agent hook (README "Multi-Turn Agent"), our executor drives terminus-2 on the Harbor server
   --train.agent_func_path "$BENCH/configs/openrlhf/w2_agent.py"
   # 65,536 = prompt + completion of one model call (one episode = one sequence); 8192 per call is set in the executor
   --data.max_len 65536
   --rollout.max_new_tokens 8192
   # 8 tasks x 8 episodes = 64 episodes per step, all in flight at once (= the 64-box cap); 60 steps from 480 prompts
   --rollout.batch_size 8
   --rollout.n_samples_per_prompt 8
   --train.batch_size 64
   --data.max_samples $((8 * ${MAX_STEPS:-60}))
   # ZeRO-3 as in OpenRLHF's 64k agent example (train_reinforce_baseline_ray_agent_async.sh): a 65k-token sample's fp32
   # logits alone are ~65 GB, and ZeRO-2's replicated params + grads (~40 GB) would not leave room for them
   --ds.zero_stage ${ZERO_STAGE:-3}
   --ckpt.save_steps ${SAVE_STEPS:-20}
   --ckpt.max_num 3
)
[ -n "${ADAM_OFFLOAD:-}" ] && W1_ARGS+=(--ds.adam_offload)
