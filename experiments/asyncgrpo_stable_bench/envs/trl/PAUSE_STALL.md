# Slow, erratic weight-sync pause in trl-w1-S-recommended-364705

Status: log analysis done (final 364705 logs, 155 syncs; shared rerun 364895). Fix written as a patch, not yet measured on GPU: the repro was still queued behind the node cap at hand-back (see "Repro" at the end). Scripts and outputs in `envs/trl/pause_stall/`.

## Short answer

The 20 s is not the engines and not DP/API-server coordination. 85% of it is the `/pause` HTTP request waiting to be handled by the vLLM API server process it landed on, while the engines keep generating at full speed. The rest is the engines' "paused" replies waiting behind the same API server's output backlog. The API servers are backlogged because the recommended arm turns on score centering, which asks `/v1/completions` for the top-8 log-probs of every token of 16k-token completions. The shared arm (no score centering, `logprobs=0`), same vLLM, same 8 API servers, 1024 in flight instead of 688, paused in 0.1 s on all 9 of its steps.

## Where the time goes (run 364705, 91 syncs, `pause_stall/timeline.py`, hopper-cpu job 364878)

`t0` = start of the `weight_sync` span (spans-train.jsonl), `t_pause = t0 + perf/weight_sync_pause_s`, PAUSED_ALL / UNPAUSED from vllm.log (1 s resolution, all four EngineCore_DPn always log within the same second).

- `pause_s`: mean 20.0, median 13.6, max 70.9 s.
- `t0 -> PAUSED_ALL`: mean 16.9 s, median 11.7 s, **85% of all pause time**. During that window the engines are not paused at all: the client keeps receiving replies at the normal rate (`replies_in_pause` up to 663 in step 10; Spearman 0.94 with `pause_s`, because generation simply goes on while the request waits).
- `PAUSED_ALL -> /pause returns`: mean 3.0 s, median 0.9 s, 15%. In the steps where it's long (13-26 s: steps 25, 37, 48, 53, 85), the client is also still receiving replies **after every engine had stopped stepping**: 104-282 "late" replies, the last one 14-27 s after PAUSED_ALL. Those completions finished on the GPU before the pause; they sat in the API servers' output path for up to 27 s. The pause's own reply from the engines (a utility output on the same ZMQ socket and event loop) waits behind that backlog.
- Weight transfer 0.2 s, barrier 0.0 s, engines go PAUSED_ALL -> UNPAUSED within ~1 s once the request gets through.

So the pause is as slow as the API server's event loop is behind, and that lag varies a lot from moment to moment and server to server: 0.04 s on some steps, 66 s on others.

## Scaling

Per step, against `pause_s` (Spearman): in-flight at `t0` +0.30 (it's pinned at 688 almost always, so there's little signal), replies/s in the previous 60 s +0.06, generated tok/s in the previous 60 s +0.12, mean completion length +0.07, completions >= 8192 tokens +0.11. Nothing in the minute before the sync predicts its pause well: the lag comes in bursts. The between-run comparison is the clear signal:

| run    | arm                           | logprobs per token | in flight | pause_s                            |
| ------ | ----------------------------- | ------------------ | --------- | ---------------------------------- |
| 364684 | shared                        | 0 (sampled only)   | 1024      | 0.0-0.1 s, all 9 steps             |
| 364705 | recommended (score centering) | top-8              | 688       | mean 20.0, median 13.6, max 70.9 s |

Same vLLM 0.33.0, same serve args, same TRL commit (5623e8b), same node type. Generated tok/s were similar (shared 93-126k at steps 3-9, recommended 38-137k).

## Shared vs recommended, side by side (coordinator's A/B, hopper-cpu job 364903)

The shared rerun 364895 (same code, vLLM 0.33, serve args, async reward) against the live recommended run 364705, whole runs so far:

|                                                                     | shared 364895 (15 steps)              | recommended 364705 (155 steps)     |
| ------------------------------------------------------------------- | ------------------------------------- | ---------------------------------- |
| `perf/weight_sync_pause_s`                                          | 0.0-0.1 s, every step                 | mean 18.2, median 13.2, max 70.9 s |
| `t0 -> PAUSED_ALL`                                                  | engines paused within the same second | mean 15.5 s (85% of the pause)     |
| steps with replies delivered >= 2 s after PAUSED_ALL                | 0                                     | 45 of 155                          |
| client concurrency (`rollout/concurrency`)                          | 1024                                  | 684                                |
| requests the engines actually run (`rollout/vllm_running_requests`) | ~1016 after ramp-up                   | **median 245, p90 425**            |
| `rollout/generated_tok_s`                                           | ~89k                                  | ~78k                               |
| trainer step (`train_step` span)                                    | 21-23 s after warm-up                 | mean 23.3 s                        |

In the recommended run `pause_s` doesn't follow the trainer: Spearman vs `train_step_s` -0.11, vs `sample/rollout_queue_size` +0.13, vs `rollout/concurrency` -0.06, vs generated tok/s +0.01. So the trainer-side arm differences (token_budget 18432 x 43 micro-batches vs 49152 x 16, std-normalized advantages, no epsilon_high) don't explain it: none of them reach vLLM. The only difference vLLM sees between the arms is the request: score centering sets `logprobs=8` on every `/v1/completions` request, the shared arm sends `logprobs=0`.

The `vllm_running_requests` row is the tell. In the recommended arm the client has ~684 requests open but the engines run a median of 245: the other ~440 are inside the API servers, either not yet handed to an engine or finished and not yet sent back. That's the backlog the `/pause` request queues behind. The shared arm, with more requests open, keeps the engines at ~1016.

## prime-rl and miles

- **prime-rl** (run 364392) uses the very same vLLM call: `POST /pause?mode=keep&clear_cache=false` (`prime_rl/orchestrator/clients.py:560`, served by its own `/pause` route that calls `pause_generation(mode="keep", clear_cache=False)`, `inference/vllm/server.py:44`), on the same topology (vLLM DP=4, 4 API servers, up to 1024 in flight). Its per-engine pause time over 201 syncs: mean 0.048 s, max 0.074 s. The difference is what the API servers do per token. prime-rl generates through `/inference/v1/generate` with its own handler (`prime_rl/inference/vllm/serving_tokens.py`), whose docstring names this exact problem: "Per-token Python objects are what makes the API server slow under RL load: ~100+ concurrent 16k-token requests keep tens of millions of them alive, and every gen-2 GC pass walks all of them. So the handler also turns off detokenization ... and asks for flat logprobs". It also returns log-probs as packed float32 arrays instead of per-token JSON.
- **miles** runs SGLang, one engine per GPU, and calls `pause_generation(mode="in_place")` on each engine directly (`miles/backends/training_utils/weight_update/session.py:16`), no shared HTTP front end in the path and no per-token top-k.

## Overlap with HANG_364684.md

None in cause. That hang is the rollout-worker child starved of the GIL by sympy in the reward thread; the timeline there also shows the step-9 pause of 364684 took ~1 s, consistent with the table above. One shared observation: both notes see the fp32 -> bf16 cast warning on every sync (the trainer sends fp32 master weights); it costs transfer bytes, not pause time.

## Root cause, in vLLM terms

- `/pause` is handled by whichever of the 8 API server processes accepts the connection (one shared listen socket). That process sends a `pause_generation` utility call to all 4 engines (`DPLBAsyncMPClient.call_utility_async` -> `call_utility_all_async`) and waits for all 4 replies on its own ZMQ output socket (`process_outputs_socket`). The engine side is instant for `mode=keep`: it sets PAUSED_ALL, has no work left, syncs the device and replies (`EngineCoreProc.pause_scheduler`, `v1/engine/core.py:2180`). No DP wave or cross-server coordination is involved (Qwen3.5-2B is dense, so the DP engines are independent `EngineCoreProc`s).
- Both waits, for the request to be read and for the engines' replies, run on that API server's single asyncio loop. When the loop is behind, the pause is behind by the same amount.
- With `logprobs=8` on `/v1/completions`, vLLM's output processor keeps, per in-flight request, one `dict[int, Logprob]` with ~9 `Logprob` objects per generated token until the request finishes (`flat_logprobs` is off for every OpenAI endpoint). At ~85 requests per API server and completions up to 16k tokens, that's on the order of 10M live objects per server. Each finished request then builds a top-8 `/v1/completions` response with string keys (`"token_id:N"`) for every token. Gen-2 GC passes over that heap and the per-response builds stall the loop for seconds at a time. The 2026-10 logprob bench (memory note) measured the same mechanism at 1k tokens: ~30% of the API-server loop in GC, pauses up to 406 ms; at 16k tokens the heap is ~16x larger.
- `logprobs=0` keeps one `Logprob` per token and the shared arm never builds up a backlog, at 1024 in flight.

## Proposed fix

A patch to TRL that returns score-centering logprobs flat from the generate endpoint (written against TRL 5623e8b; not part of this directory, and not applied to the source the runs used).

When score centering is on (`num_sampler_logprobs > 0`), the rollout worker sends its requests to `/inference/v1/generate` instead of `/v1/completions`, with `"detokenize": false, "flat_logprobs": true` in `sampling_params`. That endpoint takes vLLM's `SamplingParams` as-is, so `flat_logprobs` (engine-level only on the OpenAI endpoints) can be set by the client: the output processor then keeps a few flat lists per request instead of a dict and 9 objects per token, and returns integer token ids. This is what prime-rl does (it also patches the response into packed arrays; this patch doesn't). Details:

- `logprobs=0` (shared arm, no score centering) keeps `/v1/completions` unchanged.
- The top-k parse switches to the 0.33 generate format, `{"token_id": int, "logprob": float, "rank": int}`. `_by_token_id` goes away. This also fixes the existing replay path (`top_k`/`top_p` + score centering), which read `t["token"]`, a field `/inference/v1/generate` doesn't return on 0.33. TRL's own `parse_generate_logprobs` test (`tests/test_vllm_client_server.py`) already uses the `token_id` format.
- It needs the server started with `--enable-scale-out` (vLLM >= 0.30). The startup check now raises a clear error when score centering is on and the route is missing, instead of retrying a 403 for minutes. Score centering is on by default, so the docs (`async_grpo_trainer.md`, the `num_sampler_logprobs` docstring) and `launch/trl/w1.sbatch` (add `--enable-scale-out` via `VLLM_EXTRA`) need the flag too. Not in the patch.

No-code alternatives, weaker: `--api-server-count 16` halves the per-server heap but keeps the mechanism; a smaller `num_sampler_logprobs` cuts objects per token roughly linearly but changes the method. A different pause mode doesn't help: `wait` drains every request and `abort` throws away up to 16k tokens each, and the delay is in reaching the engine, not in what the engine does.

## Repro (queued, not yet run)

`envs/trl/pause_stall/repro.sbatch` (b200-queue waiter 364879): vLLM with the w1 serve args plus `--enable-scale-out`, loaded by `load.py` like the rollout worker (688 in flight, keep-alive pool, completion lengths like 364705), TRL's `/pause` + `/resume` every 20 s, per-process CPU/RSS sampled by `cpu.py`. Three arms on one server: `comp8` (`/v1/completions`, logprobs 8, the recommended arm), `gen8` (`/inference/v1/generate`, logprobs 8, flat, no detokenize, the patch), `comp0` (logprobs 0, the shared arm). Output in `pause_stall/logs/<jobid>/` (`load.jsonl` pause rows, `vllm.log`, `cpu.jsonl`). It would confirm: (1) `comp8` reproduces multi-second pauses on a standalone server with no trainer, (2) `gen8` brings them to the `comp0` level (~0.1 s), (3) API-server processes sit near 100% CPU and grow RSS under `comp8` but not `gen8`, (4) the `gen8` response format matches the patch's parse (the `check` row).

## Options measured (2026-11-10, job 365055)

One b200 node, `pause_stall/options.sbatch`: `repro.sbatch` three times (8, 16 and 4 API servers), every arm the same load as before (688 in flight, completion lengths like 364705, `/pause?mode=keep&clear_cache=false` + `/resume` every 20 s), 120 s fill then 600 s measured. Logs in `pause_stall/logs/365055-api{8,16,4}/`, summaries from `pause_stall/options.py` (hopper-cpu jobs 365129, 365169, 365170). Harness changes: `API_SERVERS` and `OUT_SUFFIX` knobs in `repro.sbatch`; `load.py` logs every client error (`kind: err`) and the node's kernel TCP counters per arm. The load itself is unchanged (same RNG draws), so the old arms behave as before.

| arm                        | API servers | pauses | pause mean / median / p90 / max (s) | gen tok/s                                              | API-server CPU (cores/server) | API-server peak RSS (GiB) | client errors |
| -------------------------- | ----------- | ------ | ----------------------------------- | ------------------------------------------------------ | ----------------------------- | ------------------------- | ------------- |
| comp8 (TRL default)        | 8           | 26     | 12.0 / 10.0 / 32.2 / 38.2           | 65.3k                                                  | 0.99 (pegged)                 | 3.84 (max 4.24)           | 198           |
| comp4                      | 8           | 30     | 6.7 / 5.8 / 15.4 / 19.3             | 79.9k                                                  | 0.98 (pegged)                 | 2.93                      | 8             |
| comp2                      | 8           | 30     | 0.05 / 0.04 / 0.06 / 0.21           | 91.1k                                                  | 0.85                          | 2.71                      | 0             |
| comp1                      | 8           | 30     | 0.04 / 0.04 / 0.05 / 0.06           | 91.9k                                                  | 0.65                          | 2.59                      | 0             |
| comp0 (no score centering) | 8           | 30     | 0.05 / 0.05 / 0.06 / 0.07           | 95.9k                                                  | 0.57                          | 2.54                      | 0             |
| comp8                      | 16          | 30     | 0.06 / 0.05 / 0.08 / 0.35           | 89.5k                                                  | 0.81 (13.0 total)             | 2.58                      | 0             |
| comp8                      | 4           | 6      | 71 / 11.5 / 270 / 270               | ~40.7k (client-side reply tok/s; `/metrics` timed out) | 1.00 (pegged)                 | 6.62 (max 6.80)           | 5             |

p90 here is the value at index `int(0.9 n)` of the sorted pauses. comp8 at 8 servers reproduces the earlier runs (365007: mean 14.0, max 64 s, 65.8k tok/s). Same-day comp0 is 95.9k, higher than 365007's 81.6k, so the old comp0 number was low; throughput comparisons below are against today's comp0.

What it shows:

- The stall is API-server CPU saturation. In comp8 and comp4 every API server sits at 1.0 core (its single asyncio loop is full). At comp2 they drop to 0.85 and the pause goes to the comp0 level. Halving the per-server load (16 servers) does the same for comp8; doubling it (4 servers) makes it far worse (270 s pause, about half the throughput).
- comp2 and comp1 cost 4-5% gen tok/s vs comp0 (91.1k / 91.9k vs 95.9k); comp8 at 16 servers costs 7% (89.5k) and uses 13 cores of API-server CPU instead of 6.8.
- comp4 at 8 servers is not enough: pauses halve but stay at 6.7 s mean, 19 s max.

### Connection resets

They come from the server, not the client, and they are a symptom of the stall, not a separate bug. In comp8: 187 `ClientOSError(104)` + 11 `ServerDisconnectedError`, all before the response headers (the request was sent, and no reply ever started), median request age 2.2 s, 115 of 198 within 2 s of a pause/resume. The node's `TcpExt.TCPAbortOnClose` rose by exactly 187 during the arm (1 in comp0, 0 in comp2/comp1). That counter counts a local process closing a socket that still holds unread received data, which sends an RST. The client sees "reset by peer", so the closer is the API server, closing keep-alive connections that already held the next request it hadn't read yet. The likely trigger is uvicorn's keep-alive timeout (`VLLM_HTTP_TIMEOUT_KEEP_ALIVE=5`) firing on a loop that's seconds behind. That part is inferred, not traced. vLLM logs nothing for them: the only ERROR tracebacks in `vllm.log` are one per arm end, from the harness's `/inference/v1/abort_requests`. Errors appear only in the saturated arms (comp8 198, comp4 8, 4-server comp8 5, all others 0). The rollout worker retries them (`_RETRYABLE_HTTP_ERRORS` includes `aiohttp.ClientError`), and since they fail before any generation, they cost a retry, not tokens.

### What score centering loses with fewer sampler logprobs

From `AsyncGRPOTrainer.compute_loss` (score centering branch): per position, with q the sampler's top-k probabilities and p the policy's probabilities of the same ids, `residual_j = p_j * 1[0.5 <= p_j/q_j <= 5] - 1[tail ratio in range] * p_j`, where every token outside the top-k shares one ratio `(1 - sum p) / (1 - sum q)`. The correction is `sum_j residual_j * log p_j`, i.e. the sampler expectation of the masked importance-weighted score. Two consequences:

- When no top-k token and not the tail are masked, the residual is exactly zero for any k. k only matters at positions where masking happens.
- Smaller k puts more tokens into the single tail bucket, so per-token masking outside the top-k gets averaged away. At k=1 the top-k is just the sampler's argmax, and a masked non-argmax token only shows up through the tail average.

Measured offline (`pause_stall/topk/`, hopper-dev job 365054; 48 DAPO-math prompts, vLLM samples with top-20 processed logprobs, policy log-probs from HF, the correction's gradient w.r.t. the logits computed exactly over the vocabulary, k=20 as reference). Two cases from run 364705's checkpoints: (A) sampler = policy = checkpoint-200, the mismatch being numerics only (bf16 vLLM vs HF), like a fresh weight sync; (B) sampler checkpoint-150, policy checkpoint-200, a 50-step-stale stress case.

| k   | top-k sampler mass (A) | correction / loss gradient (A) | error vs k=20 / loss gradient (A) | top-k mass (B) | correction / loss gradient (B) | error vs k=20 / loss gradient (B) |
| --- | ---------------------- | ------------------------------ | --------------------------------- | -------------- | ------------------------------ | --------------------------------- |
| 1   | 0.898                  | 1.9e-5                         | 4.2e-5                            | 0.874          | 0.094                          | 0.075                             |
| 2   | 0.962                  | 1.9e-5                         | 4.2e-5                            | 0.947          | 0.099                          | 0.052                             |
| 4   | 0.987                  | 4.3e-5                         | 7.1e-6                            | 0.979          | 0.103                          | 0.029                             |
| 8   | 0.996                  | 4.6e-5                         | 2.0e-6                            | 0.992          | 0.105                          | 0.012                             |
| 20  | 0.999                  | 4.6e-5                         | 0                                 | 0.998          | 0.105                          | 0                                 |

Norms are over all positions (root of summed squares). "Loss gradient" is the k=20 per-token gradient `weight * (e_x - p) - correction`. Masked sampled tokens: 0 in (A), 4.2% in (B). Run 364705 itself logged `sampling/masked_mean` 4e-5 on average (max 1.2e-3) and `sampling/sampler_mass` (top-8) 0.9945, so the real run is close to case (A): the correction is about 5e-5 of the gradient there whatever k is, and going from 8 to 2 changes the gradient by ~4e-5 of its norm. Only under heavy staleness does k matter (k=2 off by 5% of the gradient, k=8 by 1%). Side note: with fresh weights, score centering's correction term is nearly zero, so in this regime the recommended arm's objective is effectively masked IS (`weight * log p`, ratio in [0.5, 5]) without PPO clipping.

### Recommendation for 1.18

- Set the `num_sampler_logprobs` default to 2. It removes the stall at the w1 load (pause 0.05 s mean, 0.21 s max, same as comp0), costs ~5% gen tok/s vs no logprobs (vs -32% for top-8 today), drops the resets to 0, and changes the score-centering gradient by ~4e-5 of its norm at the real run's staleness. k=1 is no faster here (91.9k vs 91.1k) and loses more under staleness, so 2 is the better default.
- Document `--api-server-count` as the knob if someone wants k=8: 16 servers fix comp8 at this load (0.06 s mean, 89.5k), at the cost of roughly twice the API-server CPU. Don't rely on it for the default: the margin depends on in-flight count and completion length, and 4 servers show how quickly it falls over.
- Keep score centering on. Turning it off isn't needed for the stall.
- Don't land the `/inference/v1/generate` patch (365007 already showed it doesn't fix the tail).
