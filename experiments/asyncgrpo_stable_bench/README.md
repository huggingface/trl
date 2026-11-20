# AsyncGRPOTrainer stable benchmark

The cross-library RL benchmark behind the post [AsyncGRPO benchmark](https://huggingface.co/spaces/trl-lib/asyncgrpo-benchmark), run for the stable release of TRL's `AsyncGRPOTrainer` (TRL 1.18, huggingface/trl#7841). It compares TRL with prime-rl, verl, SkyRL, OpenRLHF, miles and NeMo-RL on the same models, data, rewards and hardware, each library run the way its authors recommend, with every gap written down.

This directory is the reproduction material: install scripts, configs, launchers, collectors, workload code, the failure ledger and the chart builder. The raw results of every run, in one schema, are in the HF dataset [trl-lib/asyncgrpo-stable-bench-runs](https://huggingface.co/datasets/trl-lib/asyncgrpo-stable-bench-runs).

## Layout

- `envs/<lib>/`: per-library install scripts (`install*.sh`), the resolved packages (`freeze*.txt`, from `uv pip freeze`), the pinned source commit (`SOURCE`), `setup.json` (install time and versions) and `NOTES.md`, every problem hit and how it was fixed, in the order it happened.
- `configs/<lib>/`: configs and equivalence tables (`EQUIVALENCE.md`) against the shared configs, [configs/SHARED_W1.md](configs/SHARED_W1.md) and [configs/SHARED_W2.md](configs/SHARED_W2.md).
- `launch/<lib>/`: the Slurm launchers.
- `collect/`: one collector per library (`collect/<lib>.py`) that turns its native logs into the common schema, [collect/SCHEMA.md](collect/SCHEMA.md).
- `workloads/`: the shared workloads. `math/` (W1: DAPO-Math-17k, AIME24/25 and MATH-500 evals, the reward), `agentic/` (W2: TermGrade terminal tasks through OpenEnv Harbor, terminus-2, HF Sandboxes or local Apptainer boxes; `agentic/hard/` is the harder TermGrade split of the 35B runs), `text2sql/` (miles' text-to-SQL recipe ported to TRL, used for the 35B comparison). Data files aren't checked in; each workload's `prepare.py` / `convert.py` rebuilds them.
- `results/raw/<run_id>/`: raw results in one schema for every run (`meta.json`, `steps.jsonl`, `eval.jsonl`, ...). `run_id` = `<lib>-<workload>-<size>-<tag>-<slurm job id>`.
- `results/robustness/`: the failure ledger, one row per distinct problem with its cause, fix and time lost (`failures.jsonl`, summarized in `summary.md`).
- `results/tables/`: derived tables, including [results/tables/throughput_fair.md](results/tables/throughput_fair.md), the common throughput definition.
- `figures/build.sh`: rebuilds the charts (`figures/post/` for the post, `figures/out/` for per-run diagnostics, `figures/interactive/index.html`). See [figures/README.md](figures/README.md).

## Models and workloads

| size | model | workloads |
|---|---|---|
| S | `Qwen/Qwen3.5-2B` | W1 math |
| M | `Qwen/Qwen3.5-9B` | W1 math, W2 agentic |
| 35B | `Qwen/Qwen3.5-35B-A3B` (MoE, LoRA) | W2 agentic, TermGrade-hard, text-to-SQL |

Qwen3.5 is a hybrid (gated DeltaNet + full attention). Without `flash-linear-attention` (with `fla-core`) and `causal-conv1d` it falls back to a pure-torch path that trains about 20x slower, so every venv installs both and every library checks that fla is actually used (see `envs/trl/NOTES.md`).

## Paths

The scripts hardcode the paths of our cluster (`/fsx/...`). Swap them for your own:

| placeholder | what | ours |
|---|---|---|
| `$BENCH` | this directory | `.../experiments/asyncgrpo_stable_bench` in a TRL checkout |
| `$RUNS` | big outputs: checkpoints, native logs, job logs, one dir per `<lib>/<run_id>` | `/fsx/<user>/stable-bench-runs` |
| `$SRC` | library source checkouts, at the commit in `envs/<lib>/SOURCE` | `/fsx/<user>/stable-bench-src` |
| `$ENVS` | one venv per library, `$ENVS/bench-<lib>` (never inside a repo) | `/fsx/<user>/envs` |

All models and datasets come from a shared `HF_HUB_CACHE`.

## Reproduce

1. Install a library: run `envs/<lib>/install.sh` as a Slurm job (the header of each script has the command). It checks out the library in `$SRC/<lib>` at the pinned commit (TRL installs from a TRL checkout, the commit each run used is `trl_sha` in its `meta.json`) and builds `$ENVS/bench-<lib>` with `uv`; `freeze*.txt` is what it resolved for us. Some libraries need a second venv or a container for W2 or the 35B runs (`install_w2.sh`, `install_megatron*.sh`, `install_image.sh`, `envs/trl/install_ep.sh`).
2. Prepare the data: `workloads/math/prepare.py` (W1), `workloads/agentic/prepare.py` plus `base_image/` (W2, see [workloads/agentic/README.md](workloads/agentic/README.md)), `workloads/text2sql/convert.py` (text-to-SQL, see [workloads/text2sql/README.md](workloads/text2sql/README.md)). Some libraries need their own data format first (`envs/<lib>/prep_*.sh`).
3. Train: `sbatch launch/<lib>/<workload>.sbatch` with the config from `configs/<lib>/`. Timed runs take whole nodes (`--nodes=N --gres=gpu:8 --exclusive`).
4. Evaluate checkpoints: `workloads/math/eval.sbatch` (W1), `launch/trl/w2.sbatch` with `STAGE=probe` (W2 held-out tasks), `workloads/text2sql/eval.sbatch` (text-to-SQL).
5. Collect: `collect/<lib>.py` writes `results/raw/<run_id>/` from the native logs in `$RUNS`.
6. Build the tables and charts: `figures/build.sh`.

The library-specific details, including every deviation from the shared config, are in `configs/<lib>/EQUIVALENCE.md` and `envs/<lib>/NOTES.md`.

## Cluster notes

- Our runs were on 8 x B200 nodes under Slurm, with `/fsx` as the shared filesystem and node-local `/scratch`.
- Many launchers say `b200-queue ... script.sbatch`: that's our wrapper around `sbatch` that capped how many nodes the benchmark used at once. Plain `sbatch -p <gpu partition>` with the same arguments does the same.
- Installs, data prep and log analysis ran as CPU jobs, never on the login node.
- Python environments come from `uv` only, never `pip`.
- Ports are derived from the job id, never hardcoded: `PORT=$((8000 + (SLURM_JOB_ID % 90) * 10))`.
- The W2 base image for HF Sandboxes was a private Docker Space (`aminediroHF/termgrade-base` in the configs). Build your own with `workloads/agentic/base_image/build.py --space <you>/termgrade-base` and pass it to `prepare.py --image`.

## Raw data

Every run writes `results/raw/<run_id>/` in the schema of [collect/SCHEMA.md](collect/SCHEMA.md). Raw files stay raw: no smoothing, no aggregation, and tables and figures are derived from them. They aren't checked in here: all of `results/raw/` is in the HF dataset [trl-lib/asyncgrpo-stable-bench-runs](https://huggingface.co/datasets/trl-lib/asyncgrpo-stable-bench-runs), one directory per `run_id`. Download it before building the tables or charts:

```bash
hf download trl-lib/asyncgrpo-stable-bench-runs --repo-type dataset --local-dir results/raw
```

A few tables also read native logs that aren't published: `collect/t2sql_35b.py` (the text-to-SQL 35B tables) and parts of `figures/build_data.py` (35B and W2 run details) look in `$RUNS`.
