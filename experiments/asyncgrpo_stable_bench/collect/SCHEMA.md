# Raw results schema

One directory per run: `results/raw/<run_id>/`. A field the library doesn't expose is written as `null`, never guessed. Times are seconds, and timestamps are unix epoch floats.

## meta.json

`run_id`, `lib`, `lib_version`, `lib_sha`, `trl_sha` (TRL runs), `workload`, `size`, `model`, `config_file` (path in this repo), `config_sha256`, `slurm_job_id`, `nodes` (list), `gpus_total`, `gpus_inference`, `gpus_train`, `inference_engine` (e.g. `vllm 0.32.0`), `train_backend` (e.g. `fsdp2`, `megatron`), `master_dtype`, `max_staleness`, `start_time`, `end_time`, `gpu_hours`, `status` (`completed` / `failed` / `killed`), `notes`.

## steps.jsonl (one row per optimizer step)

`step`, `time` (wall clock at step end), `step_time`, `train_compute_time`, `wait_for_rollouts_time`, `weight_sync_time`, `num_rollouts`, `gen_tokens` (completion tokens generated for this step's rollouts), `trained_tokens` (non-pad tokens through the backward), `padding_fraction`, `reward_mean`, `reward_std`, `frac_zero_adv_groups`, `completion_len_mean`, `completion_len_max`, `truncated_fraction`, `staleness_mean`, `staleness_max`, `logprob_mismatch_kl` (trainer vs inference), `grad_norm`, `lr`, `peak_mem_gb` (max over train ranks).

## eval.jsonl (one row per evaluated checkpoint, written by workloads/math/eval.py)

`step`, `time` (wall clock when the checkpoint was saved), `aime24_avg16`, `aime25_avg16`, `math500_avg1`, `mean_len`.

## spans.jsonl (optional per library, required for TRL)

`kind` (`generate` / `env` / `train_step` / `weight_sync` / `wait`), `rank` or `worker`, `gpu` (when known), `start`, `end`, `step`.

## gpu.csv

1 Hz `nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv -l 1` on every node, plus a `node` column. Launch scripts start it in the background and stop it at exit.

## setup.json (per library, in envs/<lib>/)

`install_wall_s` (clean venv to import OK), `install_steps` (manual steps beyond `install.sh`), `venv_gb`, `first_step_s` (job start to the first optimizer step, per size), `conflicts` (list of pins we had to override), `patches` (list of source patches we had to apply).

## W2 additions (agentic, configs/SHARED_W2.md)

steps.jsonl, on top of the fields above: `trained_tokens` changes meaning to the SHARED_W2 count (every training sample's full sequence, once per episode, no deduplication), `loss_tokens` (loss-mask tokens), `forwarded_tokens` (what the library ran through the model after its own prefix sharing), `prefix_share_ratio` (= trained / forwarded), and for the episodes that finished in the step's window (previous step end, this step end]: `episodes_finished`, `episode_wall_mean`, `turns_mean`, `tokens_per_episode_mean`, `failed_episodes`, `budget_episodes`, `episode_model_s_mean`, `episode_sandbox_s_mean`.

episodes.jsonl (one row per finished episode, training and eval alike): `start`, `end`, `wall_s`, `task`, `reward` (null when the verifier never scored it), `status` (`scored` / `budget_scored` / `failed`), `error`, `turns`, `completion_tokens`, `seq_tokens` (last call's prompt + completion), `prompt_tokens_sum`, `phases` (Harbor's `environment_setup`, `agent_setup`, `agent_execution`, `verifier` seconds), `model_s` (sum over agent steps of the time from the previous trajectory step to the reply), `sandbox_s` (= wall - model_s), `trial`. `collect/agentic.py` builds these from a Harbor trials dir for any library.

spans.jsonl: `kind: env` spans are whole episodes (`worker: harbor`, with `task`, `status`).
