# Figures

One command rebuilds everything (it submits itself as a CPU Slurm job and waits). `BENCH`, `PY` and `RUNS` at the top of the script point at this directory, a Python with numpy, pandas and matplotlib, and the native run outputs; override them from the environment:

```bash
experiments/asyncgrpo_stable_bench/figures/build.sh
```

It runs `collect/throughput_fair.py` and `collect/t2sql_35b.py`, then `figures/build_data.py` (all plotted numbers -> `figures/data/bench.json`), then `figures/charts.py --diagnostics` and `figures/build_interactive.py`. Runs still in flight show as "in progress"; rerun it once their files land (new W1/W2 runs, W2 probe evals, 35B W2 runs under `stable-bench-runs/trl-moe-ep/*w2*` or `prime-rl/prime-rl-w2-moe-*` are picked up automatically).

- `figures/post/NN_*.{svg,png,csv}`: the blog figures (PNG at 2x), in post order. Only runs on built-in options (no `validity` note in meta.json), no recommended arms.
- `figures/interactive/index.html`: the same charts as one self-contained page (data inlined, d3 7.9.0 from cdnjs), light and dark, hover tooltips, library toggles, linked highlighting, a data table under each chart. Template: `figures/interactive/template.html`.
- `figures/out/<workload>/`: the per-run diagnostic charts below.

## Diagnostic charts

Every figure comes from `results/raw/<run_id>/` only (schema in [../collect/SCHEMA.md](../collect/SCHEMA.md)). Download `results/raw/` from the HF dataset trl-lib/asyncgrpo-stable-bench-runs first (see the top-level README). Regenerate them in a CPU job:

```bash
cd $BENCH
sbatch --wait -p <cpu partition> -c 4 --mem=16G -t 0:30:00 -o $RUNS/figures-logs/%x-%j.out \
    --wrap "cd $PWD && uv run --no-project --python 3.12 --with matplotlib --with pandas python figures/charts.py"
```

Options: `--runs REGEX` keeps only matching run ids (e.g. `--runs 'w1-S-(shared|recommended)'` to drop smoke runs), `--timeline-run REGEX` picks the TRL run of chart (i) (default: the latest one with spans), `--timeline-start <unix time>` moves its window.

Output: one folder per workload, named after what was trained, with the plotted numbers next to it in `results/tables/<folder>/<chart>.csv`. Every chart starts with a header line giving the workload, model, data, eval and hardware (`WORKLOADS` in charts.py):

| folder | what |
|---|---|
| `figures/out/w1-S_math_qwen3.5-2B_dapo-math-17k/` | W1-S: single-turn math, Qwen3.5-2B, DAPO-Math-17k, eval AIME24/25 + MATH-500 |
| `figures/out/w1-M_math_qwen3.5-9B_dapo-math-17k/` | W1-M: the same with Qwen3.5-9B (appears once a W1-M run is collected) |
| `figures/out/w2-M_agentic_qwen3.5-9B_termgrade/` | W2-M: multi-turn agentic terminal tasks, Qwen3.5-9B, TermGrade on HF Sandboxes, terminus-2 |
| `figures/out/vllm-pause-repro_qwen3.5-2B/` | standalone vLLM pause repro (no training) |

Each workload folder gets the charts below that apply to it (c, d, e, j are W1 only, k is W2 only). Smoke runs are left out. A field that is null for a run is skipped and listed in a note under the chart. Colors are fixed per library (trl blue, prime-rl orange, verl aqua, skyrl yellow, miles pink, openrlhf green); line style tells runs of one library apart (shared solid, recommended dashed, other tags dotted/dash-dot). Steady state means steps after step 5.

| file | chart |
|---|---|
| `a_reward_vs_hours` | W1 train `reward_mean` vs hours since the run's first optimizer step, one panel per library: raw thin, EMA (alpha 0.1) from step 6 thick, other libraries' shared-arm EMA in gray behind |
| `b_reward_vs_step` | the same vs optimizer step, steps 1-5 shaded. An async library's first steps train on the first rollouts to finish, which are short and mostly solved, so step 1 reward is high and falls back once long completions arrive |
| `b2_reward_vs_gen_tokens` | the same vs cumulative `gen_tokens` (billions): libraries put different numbers of tokens in a step, so this is the fairer data axis |
| `c_eval` | AIME24 avg@16, AIME25 avg@16, MATH-500 avg@1 from `eval.jsonl`, vs step (top) and vs hours since the first step (bottom) |
| `d_step_breakdown` | three aligned panels per W1 run: tokens per step (all trained vs useful = nonzero advantage, from `results/tables/throughput_fair.csv`), mean step time split into train compute / waiting for rollouts / weight sync / other, and that split per million useful tokens, which is the comparable one |
| `e_throughput` | trained tokens/s/GPU from `throughput_fair.csv`: useful solid, zero-advantage tokens the library also trained light |
| `f_peak_memory` | peak `memory.used` of the trainer GPUs and of the inference-engine GPUs per run (roles inferred: the engine's GPUs have the highest median memory), torch `peak_mem_gb` as a marker where logged |
| `f2_memory_over_time` | `memory.used` per GPU over the run (1-min max), one panel per run, trainer GPUs colored, engine GPUs gray |
| `g_staleness_truncation` | `staleness_mean`, `staleness_max`, `truncated_fraction` vs step, raw thin + EMA thick, legend under the panels |
| `h_gpu_util` | `utilization.gpu` heatmap per GPU over time, one panel per run |
| `i_trl_timeline` | TRL spans over 5 minutes: in-flight vLLM requests on top, trainer `train_step` / `weight_sync` / `wait` lanes below |
| `j_eval_step200` | W1-S eval at step 200, shared vs recommended per library, base model line |
| `k_w2_tokens` | W2 version of `d_step_breakdown`: trained tokens (every model call as its own sample) vs forwarded (after the library's prefix sharing), step time split, and step time per million trained tokens; each library's shared run when collected, else its smoke run, step 1 left out |
| `l_pause_options` | vLLM pause latency and generation throughput per `num_sampler_logprobs` (standalone repro, job 365055) |
| `field_coverage.csv` (table only) | fraction of non-null values of every `steps.jsonl` field per run, to spot collector gaps |
