# Robustness ledger

Every failure, unclear setup step and log error hit in the benchmark, per library. TRL is held to the same rules as everyone else, and our own mistakes are kept (as `ours`) so they can be excluded rather than hidden.

Files:

- `failures.jsonl`: the ledger, one row per distinct problem. Hand-maintained.
- `scan_logs.py` / `scan_logs.sbatch`: scan every run's logs under `/fsx/amine_dirhoussi/stable-bench-runs/` and write `log_errors.csv`, `log_messages_top.csv` and `log_files.csv`.
- `summary.py`: validates the ledger and writes `summary.csv` and `summary.md`.

## Adding a row

Append one JSON object per line to `failures.jsonl`. Ids are `<lib>-NNN` (next free number for that lib). All fields are required:

| field | values |
|---|---|
| `id` | `trl-011`, `all-015`, ... |
| `date` | UTC from the source: `2026-11-09T17:12Z`, or `2026-11-09` when no time is known |
| `lib` | `trl`, `prime-rl`, `verl`, `skyrl`, `miles`, `openrlhf`, or `all` (cluster facts and shared tooling that hit every lib) |
| `phase` | `install`, `setup`, `smoke`, `recipe`, `w1`, `eval`, `harness` |
| `category` | `install_break` (resolution/build/404/ABI), `missing_dep`, `docs_or_example_mismatch` (examples or docs that don't work as written, stale flags), `unclear_setup` (needed reading source to configure, or no knob for it), `config_trap` (a setting or default that silently hurts), `crash`, `hang`, `silent_correctness` (wrong numbers without an error), `perf_trap`, `cluster_unfriendly` (fixed ports, pkill, ray stop, non-empty run dir), `our_harness`, `infra` |
| `severity` | `blocker` (no valid run until fixed or worked around; every silent-correctness default counts), `major` (lost a job or a run, or changed the config), `minor`; `info` for every `observation` row (and only those) |
| `attributable_to` | `lib`, `upstream_dep`, `ours`, `infra`, `unknown`. `our_harness` rows must be `ours` and `infra` rows must be `infra`; `unknown` goes with `kind: failure_unattributed` only |
| `upstream_dep` | name of the dependency when `attributable_to` is `upstream_dep`, else `null` |
| `symptom`, `root_cause`, `fix` | one or two plain sentences each |
| `patched_lib_source` | `true` if we changed the library's behaviour in code (a file edit, `sed`, or a runtime monkeypatch); config and env vars don't count |
| `upstream_reportable` | `true` if it's worth an issue or PR upstream (not already fixed there) |
| `time_lost_min` | rough estimate of wall time lost, `null` if unknown |
| `source` | where it's documented: `envs/<lib>/NOTES.md #N`, or `report.md L<line>` / `audit/...` / `patches/...` (our internal run log, review notes and patches, not published) |
| `kind` | what the row counts as, see below. Only `failure` rows count in the headline |
| `retag_note` | one short line on why the row got its `kind` |

Then run the summary (it fails loudly on an invalid row).

## `kind`

Applied the same way to every library:

- `failure`: the library itself broke, hung, crashed, gave wrong numbers, or its own docs, examples, defaults, lock or shipped image didn't work as written. The only kind in the headline. A hang with unknown root cause stays here when none of our code runs in its path (openrlhf-008).
- `failure_unattributed`: a failure whose root cause isn't established while our harness code runs in the failing path (trl-011: our `loop_cls` override runs in the hung rollout child). Set `attributable_to: "unknown"`. Shown in its own column.
- `environment`: a third-party dependency or toolchain problem any library on this stack (Qwen3.5 on B200, torch 2.11-2.14, cu130) would hit: causal-conv1d/flash-attn/FlashInfer wheels and builds, uv cache reuse, antlr4 vs omegaconf, nvcc/g++ missing on CPU nodes, fla/Triton. If the library's own pinned lock, URL or image is what broke (verl-006, miles-006), it's `failure`. A library that ships a working lock for these gets no row.
- `observation`: warnings that didn't break anything (recovered retries, cosmetic log lines, dtype casts with no measured impact), documented limitations or hazards we didn't hit, and API/design notes (no switch for X). Severity `info`.
- `ours`: our harness, configs and agents (`attributable_to: ours`).
- `infra`: spot preemption, node failures, cluster limits (`attributable_to: infra`).

TRL is the subject of the benchmark, so its logs were read more closely than the others'. Observation counts aren't comparable across libraries; only `failure` counts are.

## Rerunning

From a login node (the jobs run on `hopper-cpu`, `--qos=low`, 4 CPUs; both are idempotent and safe to requeue):

```bash
cd /fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/results/robustness
sbatch --wait scan_logs.sbatch                            # scan all logs, then summary
sbatch --wait --export=ALL,MODE=summary scan_logs.sbatch  # summary only, after editing failures.jsonl
```

Job logs go to `logs/`. The scanner reads every `*.log`, `*.out`, `*.err` under each lib dir (and `queue/`, `status/`, `tests/` as lib `harness`), skips checkpoint, model and data dirs, and reads only the first and last 256 MB of files over 2 GB (`log_files.csv` marks them `sampled`). A file inside a run dir, or whose name carries that run's Slurm job id, is attributed to the run; anything else gets `<lib>:<file stem>` as its run id. Each matching line counts once, under the first class that matches: traceback (counted once, at the exception line), oom, cuda_error, nccl_error, timeout, retry/disconnect, killed/signal, nan_inf, warning_deprecation, error_other (ERROR-level lines and `...Error`/`Failed` lines that aren't warnings), warning_other. `error_other` is an extra class beyond the original list, for errors that fit none of the others. Messages are normalized (numbers, hex, ids, timestamps stripped) for `log_messages_top.csv`. W1 log totals in the summary cover run ids matching `-w1-S-(shared|recommended)-<jobid>`.
