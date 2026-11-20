# W1-M launch commands (Qwen3.5-9B)

Run from a background shell. Every command goes through `b200-queue` with `PRIO_TAG=PRIO_TRAIN`. Status as of 2026-11-10 12:25 UTC. Per-library M changes are in `configs/<lib>/EQUIVALENCE.md` under "M", and problems are in `envs/<lib>/NOTES.md` under "W1-M".

```bash
BENCH=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
```

## TRL: ready (smoke 365362 COMPLETED)

~79 s/step, peak reserved 107 GB per trainer rank, 120 GB per checkpoint (~2.8 min to write), ~4.8 h for 200 steps.

```bash
cd $BENCH && SIZE=M ARM=shared PRIO_TAG=PRIO_TRAIN /fsx/amine_dirhoussi/bin/b200-queue --nodes=1 --gres=gpu:8 --exclusive -t 08:00:00 launch/trl/w1.sbatch
cd $BENCH && SIZE=M ARM=recommended PRIO_TAG=PRIO_TRAIN /fsx/amine_dirhoussi/bin/b200-queue --nodes=1 --gres=gpu:8 --exclusive -t 08:00:00 launch/trl/w1.sbatch
```

The recommended arm sources `configs/trl/w1-M-recommended.env`: 49152 x 16 batching and `NUM_SAMPLER_LOGPROBS=2`.

## prime-rl: smoke pending (holds smoke slot 1 since 12:22 UTC)

Configs `w1-M.toml` and `w1-M-recommended.toml`. The shared arm still uses the custom `bench_loss.ppo_clip_loss`, mirroring S. That conflicts with the new no-custom-loss rule, so decide before launching. If S drops it, delete `[trainer.loss]` from `w1-M.toml` too. Use the command and `-t` from the prime-rl agent's hand-back once the smoke finishes. The form is:

```bash
cd $BENCH && SIZE=M ARM=shared PRIO_TAG=PRIO_TRAIN /fsx/amine_dirhoussi/bin/b200-queue --nodes=1 --gres=gpu:8 --exclusive -t <from smoke> launch/prime-rl/w1.sbatch
```

## SkyRL: smoke pending (holds smoke slot 2 since 11:35 UTC)

Built-in losses only: the shared arm is `rollout_is` and the recommended arm is dppo + prompt_mean, both through `skyrl_w1_main`. Compared with S, the overrides only change the model path and add resumable checkpoints (`ckpt_interval=50`, `max_ckpts_to_keep=1`, `resume_mode=latest`). Take the command and `-t` from the SkyRL agent's hand-back.

## OpenRLHF: smoke pending (waiting for a smoke slot)

Configs `w1-M.sh` and `w1-M-recommended.sh`. Take the command and `-t` from the OpenRLHF agent's hand-back.

## verl: blocked, not smoked

Under the no-patch rule, V1 separate_async with `hybrid_engine=False` crashes at init as shipped, and a separate agent is choosing an unpatched mode. `configs/verl/w1-M*.yaml` only changes `model.path` from S, and they need adapting to whichever mode that agent picks. `launch/verl/w1.sbatch` takes `SIZE=M`, `CONFIG=w1-M-recommended`, `MAX_STEPS` and `SAVE_STEPS`, and sets `expandable_segments` for M.

## miles: blocked, not smoked, conversion not run

The FA4 window fix is now off by default, which gives a wrong forward on B200, and a separate agent owns miles. `configs/miles/w1-M*.args` use miles' stock `scripts/models/qwen3.5-9B.py` with TP=2 + SP, DP=2 and full recompute. The first `SIZE=M` job converts HF to torch_dist into `/fsx/amine_dirhoussi/stable-bench-runs/miles/ckpt/Qwen3.5-9B_torch_dist`. `launch/miles/w1.sbatch` takes `SIZE=M`, `MAX_STEPS` and `SAVE_STEPS`. Separately, `configs/miles/w1_reward.py` now calls `reward_safe` (SHARED_W1 rev. 7). That applies to S reruns too.
