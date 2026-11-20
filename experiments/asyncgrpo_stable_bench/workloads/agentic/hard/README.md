# A harder agentic workload for Qwen3.5-35B-A3B

Started Thu Nov 12 00:42 UTC 2026 (`date -u`). Goal: a task set where the 35B's train reward climbs clearly. On W2 (28 TermGrade tasks picked for the 9B) the 35B starts at 0.76 train and moves 0.73 -> 0.77 over 60 steps (run 371885); target base pass rate on the new train tasks is ~0.15-0.45.

## Choice: a harder slice of the TermGrade pool (option b)

Same pipeline as W2 local, only a different task directory: OpenEnv Harbor server + terminus-2 host-side, local Apptainer boxes from the one `termgrade-base.sif`, same budgets (600 s, 30 calls, 8192 tokens per call, 64k context), same verifier reward.

Why this over the alternatives:

- **TermGrade (b): zero new code.** Every TermGrade task uses the same base image, so any task of the 1,004 runs on the existing `.sif` (and the existing podman tar for prime-rl). `prepare.py --image <sif>` writes the task dirs; the launchers take `SPLIT=<dir>`. The pool has room: of the 423 `unseen` fast-setup tasks, Qwen3.6-27B's published terminus-2 pass rate is 0 on 74, in [1/8, 5/8] on 96, 6/8 on 31 (`task_summary.csv`). The W2 split drew from the easy end (128 probe candidates: 64 that Qwen3.6-27B always solves, 64 at 50-88%).
- **Terminal-Bench 2.0 (a): rejected for now.** 89 tasks, each with its own Docker image (many multi-GB), so every task needs a pull + `.sif` conversion (plus podman for prime-rl), and several need hardware or long builds that local unprivileged boxes may not run. Too few tasks to filter to >= 46 in band, and a day of image work before the first probe.
- **prime-rl hub / verifiers env (c): rejected.** No published 30B-A3B-class multi-turn reward curve found quickly that maps onto our harness, and it would be a new environment for TRL (not Harbor).

## Candidate pool

`workloads/agentic/hard/select_pool.py` (hopper-cpu job): fast-setup `unseen` tasks, minus the W2 train/eval split and the known nop-pass tasks, minus tasks the 9B already solved >= 0.5 in probe 364979, with Qwen3.6-27B pass in [1/8, 6/8], plus 40 random tasks it never solved. Prepared into `stable-bench-src/termgrade/harbor-local-sif/hardprobe` (`workloads/agentic/hard/prep.sbatch`), which also runs Harbor's nop and oracle agents on local Apptainer boxes (nop must fail, oracle must pass).

## Measuring the 35B

`launch/trl/w2.sbatch STAGE=probe SANDBOX=local SPLIT=hardprobe EVAL_MODEL=Qwen/Qwen3.5-35B-A3B SAMPLES=4 HARBOR_SHARDS=4 TERMINUS_MAX_TURNS=1` (the training harness: 4 Harbor shards, terminus-2 stops at 30 calls), 4 GPUs, local boxes, 64 in flight. Split rule (`workloads/agentic/hard/split.py`): >= 3 scored episodes, pass in [0.1, 0.65], nop fails, oracle passes; 14 held out, the rest train. Written to `harbor-local-sif/hard-{train,eval}` (TRL) and `harbor-local/hard-{train,eval}` (prime-rl, podman image).

## Log
- 2026-11-12 00:59:30 UTC: pool 152 candidates (`workloads/agentic/hard/candidates.txt`, Qwen3.6-27B pass in [1/8, 6/8]: 112, plus 40 of the 74 it never solved), prepared into `harbor-local-sif/hardprobe` (job 372788, hopper-cpu). Local Apptainer validation on the CPU node: nop passes 11 (excluded), oracle fails 8 (excluded; local box only, not checked on HF), environment setup p50 15 s. Probe 372792 (waiter 372787) started 00:51 on ip-10-53-5-35 (4 GPUs, vLLM DP=4, 64 in flight, 4 Harbor shards, terminus-2 max_turns).
- 2026-11-12 01:55:02 UTC: **probe 372792 COMPLETED** (1 h 02 min, vLLM ready 694 s): Qwen3.5-35B-A3B on 152 candidates x 4, local boxes, training harness. 608 episodes, 2 unscored, **mean pass 0.254**. Per task (151 with >= 3 scored): 75 at 0, 34 at 0.25, 15 at 0.5, 18 at 0.75, 9 at 1.0.
- Split (`workloads/agentic/hard/split.py --hi 0.75`, nop-pass and local-oracle-fail tasks dropped): 58 in band -> **44 train (base pass 0.42 on the probe), 14 held-out (0.45)**, lists in `workloads/agentic/hard/{train,eval}_tasks.txt`, stats `workloads/agentic/hard/split_stats.json`, task dirs `harbor-local-sif/hard-{train,eval}` and `harbor-local/hard-{train,eval}`. [0.1, 0.5] would give only 29 train (mean 0.32), below the 32 asked for, so the 0.75 tasks are in. The probe numbers are selection-biased upward-to-band (4 samples), so an independent base eval of hard-eval is queued.
- Queued 01:54-01:57 UTC:
  - TRL 35B run, waiter **372821**: `launch/trl/moe_w2.sbatch` with `SPLIT=hard-train TAG=hard-st8 MAX_STALENESS=8 MAX_STEPS=60 SAVE_STEPS=20 HARBOR_SHARDS=4 TERMINUS_MAX_TURNS=1`, exclusive node, LoRA r32, EP=4 x FSDP=4, bf16, in-job watchdog. Run id `trl-moe-w2-Qwen3.5-35B-A3B-ep4-bfloat16-lora32-hard-st8-<job>`.
  - Checkpoint evals: hopper-cpu 372823 (`stable-bench-runs/status/moe_w2_hard_evals.sh 372821 hard-st8 "20 40 60"`): `launch/trl/w2.sbatch STAGE=probe SANDBOX=local SPLIT=hard-eval EVAL_MODEL=<merged dir>`, 4 GPUs, 4 Harbor shards, max_turns, as each merge lands.
  - Base held-out eval on hard-eval (same path, `EVAL_MODEL=Qwen/Qwen3.5-35B-A3B`): waiter 372824.
  - prime-rl 35B run, waiter **372825**: `launch/prime-rl/w2.sbatch` with `SIZE=moehard WEIGHT_BROADCAST=auto KEEP_ADAPTERS="20 40 60"` (`configs/prime-rl/w2-moehard.toml` = w2-moe.toml with dataset `termgrade-hard-train`; the launcher now also links `~/.cache/harbor/termgrade-hard-{train,eval}`). Staleness = prime-rl default 8. Its hf/step_N evals: hopper-cpu 372828 (`status/prl_w2_hard_evals.sh`, after the run ends).
- Held-out evals here use local boxes (no paid sandboxes), unlike the W2 evals on HF Sandboxes, so compare them only to the hard-eval base eval.
