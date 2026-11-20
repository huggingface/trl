# W2: agentic terminal tasks (TermGrade on HF Sandboxes, terminus-2)

The spec every library follows is [configs/SHARED_W2.md](../../configs/SHARED_W2.md). This directory holds what turns TermGrade into Harbor tasks that run on HF Sandboxes, the checks on them, the base-rate probe / eval, and the client code the TRL run shares.

## Pipeline

All of it runs on hopper-cpu except the probe (GPU, through `launch/trl/w2.sbatch STAGE=probe`).

1. Download: `ai-and/termgrade-environments` (42 MB) and the metadata of `ai-and/termgrade-trajectories` into `/fsx/amine_dirhoussi/stable-bench-src/termgrade/` (see `survey.py` for what's in them; it writes `task_summary.csv`: split, oracle flag, per-model pass rates).
2. Base image: `base_image/build.py` builds `base_image/Dockerfile` (TermGrade's base image verbatim + tmux/asciinema for terminus-2) as the private Docker Space `aminediroHF/termgrade-base`. HF Sandboxes pull it as `hf.co/spaces/aminediroHF/termgrade-base` (~6 s to a running sandbox).
3. Candidates: `select_probe.py` -> `probe_tasks.txt` (128 tasks of the `unseen` split, setup.sh without apt/curl/wget/systemctl, stratified on Qwen3.6-27B and gemma-4-31B pass rates).
4. Harbor tasks: `prepare.py --tasks <list> --out /fsx/amine_dirhoussi/stable-bench-src/termgrade/harbor/<split>` writes one Harbor task dir per task: `docker_image` = the Space, `environment/{setup.sh,prepare.sh}` uploaded to `/app`, and an `[environment.healthcheck]` that runs `prepare.sh` once before the agent starts (= the per-task half of TermGrade's Dockerfile: setup.sh, then delete the tests, the solution and the scripts). `network_mode` is dropped (public): HF Sandboxes can't disable internet.
5. Validation: `validate.sbatch` (SPLIT=<split>) runs Harbor's `oracle` agent (expect 1.0) and `nop` agent (expect 0.0) on every task. On the 128 candidates: oracle 128/128, nop passes 15 (`nop_pass_tasks.txt`: their setup.sh already produces what the tests check), which are excluded everywhere.
6. Base pass rates: `launch/trl/w2.sbatch STAGE=probe PROBE_SIZES="M S"` serves each model with vLLM and runs `probe.py` (4 episodes per task, the SHARED_W2 harness, budgets and sampling) through the same OpenEnv Harbor server TRL trains on.
7. Split: `select_split.py --probe <run>/probe-M.jsonl` -> `train_tasks.txt` / `eval_tasks.txt` (pass rate in [0.2, 0.8] with >= 3 scored episodes), then `prepare.py` once per list into `harbor/train` and `harbor/eval`.

## Files

| file | what |
|---|---|
| `survey.py` | what the tasks need (one shared Dockerfile, setup.sh features, network mode) and per-model pass rates, episode lengths and phase times from the trajectories release |
| `base_image/` | the Space Dockerfile and its builder |
| `select_probe.py`, `prepare.py`, `select_split.py` | candidate pool, TermGrade -> Harbor conversion, train/eval split |
| `validate.sbatch`, `validate_summary.py` | oracle / nop validation with Harbor's CLI on HF Sandboxes |
| `termgrade_env.py` | shared client code: `TermGradeSessionFactory` (OpenEnv client keepalive 60/600 s, message timeout 3600 s), the sampling policy, and `RecordedHarborSession`, which writes one `episodes.jsonl` row per finished episode (wall time, Harbor phase times, turns, tokens, reward, error) |
| `probe.py` | base pass rate / eval through the OpenEnv Harbor server |
| `sweep_sandboxes.py`, `sweep.sh` | list / cancel running sandboxes on the W2 image; every W2 job sweeps on exit unless another W2 job is still running |

## Sandboxes cost money and leak

A killed driver does not tear down the sandboxes its episodes started. Every W2 job (training, probe, validation) runs `sweep.sh` on exit; after any crash or `scancel`, run `sweep_sandboxes.py` (list) and `--cancel` once no W2 job is running. Concurrency is capped at 64 live sandboxes per run.

## For other libraries

The shared input is the prepared task dirs (`/fsx/amine_dirhoussi/stable-bench-src/termgrade/harbor/{train,eval}/tasks/`), which any Harbor `Trial` can run with `environment.type = hf-sandbox` and agent `terminus-2`, given Harbor >= 0.21 (hf-sandbox backend; the conversion also relies on task healthchecks, in Harbor since v0.4.0, and on uploading `environment/` for prebuilt-image tasks, since v0.9.0) and the tar patch `harbor-0.25.0-hf-sandbox-no-same-owner.patch` in this directory (written against the PyPI 0.25.0 wheel). `collect/agentic.py` turns a Harbor trials directory into the W2 `episodes.jsonl`.
