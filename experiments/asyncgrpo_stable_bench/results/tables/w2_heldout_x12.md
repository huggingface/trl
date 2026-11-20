# W2 held-out evals at 12 samples per task

## Log

- 20:05 UTC: started. The post's held-out evals (e.g. 365489, 372836) all ran on HF Sandboxes (no SANDBOX in launch.env = the probe default hf), 4 samples per task. Backend for the re-evals: local Apptainer boxes, the setup of the 12-sample hard-eval re-evals 373047/373048 (SANDBOX=local, W2_SANDBOX=singularity, HARBOR_SHARDS=4, TERMINUS_MAX_TURNS=1), ~22 min per eval, no sandbox cost or orphan risk. Bases are rerun in the same set.
- 20:09 UTC: driver 373564 on hopper-cpu (stable-bench-runs/queue/x12/driver.sh): at most 4 evals in flight through b200-queue (-J w2local-evalx12-<key>, --gres=gpu:4 -t 4:00:00, PRIO_TAG=PRIO_TRAIN, SAMPLES=12, 35B with VLLM_ENGINE_READY_TIMEOUT_S=2700), order bases, step 60, 20, 40; skips a key whose TAG=x12-<key> dir finished, 2 attempts max. First driver 373563 cancelled before it started (its done-check used a grep backreference; replaced with awk).
- 20:09 UTC: figures/build_data.py now prefers a 12-sample eval (TAG contains x12 or SAMPLES=12) over a 4-sample one of the same checkpoint and split; episodes_per_eval comes from the data.
- 20:34 UTC: first round (373569-373572) ran ~25 min each, vLLM ready in ~11-12 min, 168 episodes per eval with 0-2 unscored (verifier timeouts at 300 s on eval[1]). Prints: base 9B 0.301, TRL 9B step 60 0.443, prime-rl 9B step 60 0.411.
- 20:50 UTC: coordinator hold for SkyRL t2sql 373617. Driver 373564 replaced by 373623, which submits nothing while queue/HOLD names a job that is still PENDING, and removes the flag once it runs. Cancelled waiters 373600/373601/373604 (verl9b-60, trl35b-60, prl35b-60; blocked on the node cap, no GPU job yet) so they could not take the slot; their attempt counts reset to 0. trl9b-20 (373606) keeps running. Finished so far: base9b 373569, base35b 373570, trl9b-60 373571, prl9b-60 373572.
- 21:07 UTC: SkyRL 373617 running, hold released at 21:00. Driver bug: when hopper-cpu took >60 s to accept a waiter, the driver recorded an empty waiter id and queued a duplicate (prl35b-60: 373639 and 373645). Cancelled duplicate 373645 before it submitted anything; driver now waits for the id (or for the submit to exit), running as 373648. Coordinator capped b200 at 2 nodes for ~30-60 min and holds verl9b-60 (373637) on purpose: left as is. trl9b-20 (373606) finished.
- 00:26 UTC: driver 373648 done, all 17 evals finished (last at ~00:26). Building the table and figures.
- 00:28 UTC: table written (collect/w2_heldout_x12.py, job 373845) and figures rebuilt with the x12 evals preferred. Binomial SE fell from ~0.066 to ~0.038 as planned, but the task-clustered SE is still 0.06-0.11: with 14 tasks it is dominated by between-task spread, which more samples per task cannot shrink.

TermGrade held-out split (`SPLIT=eval`, 14 tasks), terminus-2, 30 model calls and 600 s per episode, thinking off. Every row ran on **local Apptainer boxes** (`SANDBOX=local`, `HARBOR_SHARDS=4`, `TERMINUS_MAX_TURNS=1`), 12 episodes per task = 168 per eval. The post's numbers came from **HF Sandboxes** at 4 episodes per task (56 per eval) without `TERMINUS_MAX_TURNS`. The backend and the turn cap changed, so the bases were rerun too: these numbers replace the HF-Sandbox ones only as a set, never row by row.

SE (binomial) = sqrt(p(1-p)/n) over scored episodes. SE (task) = standard error of the mean of the 14 per-task means, which counts episodes of one task as correlated. Δ base = mean minus the same-size base model in the same set.

## 12-sample results (local boxes)

| model | run | step | mean | scored / episodes | SE (binomial) | SE (task) | Δ base |
|---|---|---|---|---|---|---|---|
| 9B | base | - | 0.301 | 166 / 168 | 0.036 | 0.071 |  |
| 9B | TRL (st8) | 20 | 0.386 | 166 / 168 | 0.038 | 0.073 | +0.084 |
| 9B | TRL (st8) | 40 | 0.417 | 168 / 168 | 0.038 | 0.069 | +0.115 |
| 9B | TRL (st8) | 60 | 0.443 | 167 / 168 | 0.038 | 0.067 | +0.142 |
| 9B | prime-rl | 20 | 0.337 | 166 / 168 | 0.037 | 0.072 | +0.036 |
| 9B | prime-rl | 40 | 0.405 | 168 / 168 | 0.038 | 0.087 | +0.104 |
| 9B | prime-rl | 60 | 0.411 | 168 / 168 | 0.038 | 0.084 | +0.110 |
| 9B | verl | 20 | 0.369 | 168 / 168 | 0.037 | 0.063 | +0.068 |
| 9B | verl | 40 | 0.370 | 165 / 168 | 0.038 | 0.073 | +0.068 |
| 9B | verl | 60 | 0.429 | 168 / 168 | 0.038 | 0.066 | +0.127 |
| 35B-A3B | base | - | 0.577 | 168 / 168 | 0.038 | 0.105 |  |
| 35B-A3B | TRL (st8) | 20 | 0.595 | 168 / 168 | 0.038 | 0.098 | +0.018 |
| 35B-A3B | TRL (st8) | 40 | 0.554 | 166 / 168 | 0.039 | 0.093 | -0.023 |
| 35B-A3B | TRL (st8) | 60 | 0.607 | 168 / 168 | 0.038 | 0.078 | +0.030 |
| 35B-A3B | prime-rl | 20 | 0.554 | 168 / 168 | 0.038 | 0.106 | -0.024 |
| 35B-A3B | prime-rl | 40 | 0.560 | 168 / 168 | 0.038 | 0.105 | -0.018 |
| 35B-A3B | prime-rl | 60 | 0.583 | 168 / 168 | 0.038 | 0.098 | +0.006 |

## Against the post's 4-sample HF-Sandbox numbers

| model | run | step | old mean (56 ep.) | old SE (bin / task) | new mean (168 ep.) | new SE (bin / task) | new - old | old Δ base | new Δ base |
|---|---|---|---|---|---|---|---|---|---|
| 9B | base | - | 0.370 | 0.066 / 0.084 | 0.301 | 0.036 / 0.071 | -0.069 |  |  |
| 9B | TRL (st8) | 20 | 0.382 | 0.066 / 0.088 | 0.386 | 0.038 / 0.073 | +0.004 | +0.011 | +0.084 |
| 9B | TRL (st8) | 40 | 0.464 | 0.067 / 0.078 | 0.417 | 0.038 / 0.069 | -0.048 | +0.094 | +0.115 |
| 9B | TRL (st8) | 60 | 0.429 | 0.066 / 0.100 | 0.443 | 0.038 / 0.067 | +0.015 | +0.058 | +0.142 |
| 9B | prime-rl | 20 | 0.411 | 0.066 / 0.104 | 0.337 | 0.037 / 0.072 | -0.073 | +0.040 | +0.036 |
| 9B | prime-rl | 40 | 0.500 | 0.067 / 0.087 | 0.405 | 0.038 / 0.087 | -0.095 | +0.130 | +0.104 |
| 9B | prime-rl | 60 | 0.482 | 0.067 / 0.100 | 0.411 | 0.038 / 0.084 | -0.071 | +0.112 | +0.110 |
| 9B | verl | 20 | 0.400 | 0.066 / 0.072 | 0.369 | 0.037 / 0.063 | -0.031 | +0.030 | +0.068 |
| 9B | verl | 40 | 0.418 | 0.067 / 0.093 | 0.370 | 0.038 / 0.073 | -0.048 | +0.048 | +0.068 |
| 9B | verl | 60 | 0.464 | 0.067 / 0.101 | 0.429 | 0.038 / 0.066 | -0.036 | +0.094 | +0.127 |
| 35B-A3B | base | - | 0.518 | 0.067 / 0.081 | 0.577 | 0.038 / 0.105 | +0.060 |  |  |
| 35B-A3B | TRL (st8) | 20 | 0.589 | 0.066 / 0.097 | 0.595 | 0.038 / 0.098 | +0.006 | +0.071 | +0.018 |
| 35B-A3B | TRL (st8) | 40 | 0.564 | 0.067 / 0.116 | 0.554 | 0.039 / 0.093 | -0.009 | +0.046 | -0.023 |
| 35B-A3B | TRL (st8) | 60 | 0.607 | 0.065 / 0.113 | 0.607 | 0.038 / 0.078 | +0.000 | +0.089 | +0.030 |
| 35B-A3B | prime-rl | 20 | 0.589 | 0.066 / 0.110 | 0.554 | 0.038 / 0.106 | -0.036 | +0.071 | -0.024 |
| 35B-A3B | prime-rl | 40 | 0.518 | 0.067 / 0.093 | 0.560 | 0.038 / 0.105 | +0.042 | +0.000 | -0.018 |
| 35B-A3B | prime-rl | 60 | 0.625 | 0.065 / 0.094 | 0.583 | 0.038 / 0.098 | -0.042 | +0.107 | +0.006 |

Eval jobs (new / old): base9b 373569 / 365489, trl9b-20 373606 / 372718, trl9b-40 373791 / 372768, trl9b-60 373571 / 372785, prl9b-20 373679 / 370583, prl9b-40 373806 / 372305, prl9b-60 373572 / 371700, verl9b-20 373746 / 372154, verl9b-40 373826 / 372207, verl9b-60 373637 / 372379, base35b 373570 / 372068, trl35b-20 373756 / 372719, trl35b-40 373828 / 372774, trl35b-60 373681 / 372836, prl35b-20 373761 / 372600, prl35b-40 373833 / 372662, prl35b-60 373680 / 372663
