# W2 step time: TRL vs prime-rl

Written 2026-11-11 09:46 UTC. Runs: prime-rl `prime-rl-w2-M-shared-368590` (local podman), TRL `trl-w2-M-local-366992` (local Apptainer, still running, read at step ~54) and `trl-w2-M-shared-364987` (HF Sandboxes). Numbers come from `stable-bench-runs/status/w2_step_time_analysis.py` (hopper-cpu job 370138, output `status/w2_step_time.json`), a `/metrics` snapshot of 366992's vLLM (`status/w2_local_vllm_metrics_0916.txt`) and `top` on 366992's node. "Steady" means episodes that started after step 1.

## Short answer

TRL's steps are slow because its episodes take 3x longer to finish (548 s vs 186 s at the same 62 in flight), not because it does more work per episode or because vLLM is slow. Both libraries run the same episode shape: ~23-24 calls, ~15k sampled tokens, ~640 tokens per call. In TRL, vLLM spends only 71 s per episode on them (3.1 s per call). Most of the rest goes to one Python thread: the OpenEnv Harbor server runs all 64 terminus-2 agents, litellm and the capture proxy on a single asyncio loop, and that thread sits at ~96% of a core for the whole run. Every model call, tmux read and verifier exec waits behind it.

Step time follows from rollout throughput in both runs: step ≈ episodes consumed per step x episode wall / episodes in flight. TRL local: 67.9 x 548 / 61.9 = 601 s (measured 588). prime-rl: 62.6 x 186 / 62.1 = 187 s (measured 185). Training is never the critical path in either run: both spend most of the step waiting for rollouts.

## Component by component

Episode rows are means over steady, scored episodes. Per-call times are per captured model call.

| | prime-rl 368590 | TRL local 366992 | TRL HF 364987 |
|---|---|---|---|
| step time (mean / median) | 185 / 181 s | 588 / 594 s | 496 / 521 s |
| rollout wait per step | 120 s | 441 s | 344 s |
| fwd+bwd per step | 62 s | 96 s | 100 s |
| weight sync per step | 1.6 s | 0.3 s | 0.3 s |
| episodes consumed per step | 62.6 (54 shipped after zero-adv pruning) | 67.9 | 56.8 |
| episodes in flight (time-weighted) | 62.1 | 61.9 | 61.5 |
| episodes finished per second | 0.333 | 0.113 | 0.113 |
| episode wall | 186 s | 548 s | 546 s |
| model calls per episode | 24.4 | 22.9 | 23.1 |
| sampled tokens per episode | 15.7k | 14.4k | 15.5k |
| sampled tokens per call | 643 | 630 | 671 |
| final context | 31.6k | 30.3k | 34.5k |
| time in vLLM per call | 4.5 s (interception span, ~37 requests in vLLM) | **3.1 s** (vLLM e2e latency, ~8 requests in vLLM, 4.7 ms per token, 0.1 s prefill, 0.2 ms queue) | not recorded |
| time in vLLM per episode | 112 s | **71 s** | not recorded |
| call-to-call gap minus the agent's requested wait | 4.5 s model + ~0.9 s tmux I/O | **16.1 s** | 15.2 s |
| agent's requested command waits | inside harness time (61 s total) | 34 s | 33 s |
| agent phase, first 30 calls | 173 s | 411 s | 388 s |
| calls past the 30-call budget (proxy's canned stop) | 0: episode ends at call 30 | 8.9 per episode, 34 s per episode (120 s on the 28% that hit it) | 8.0 per episode, 38 s |
| box start + setup | 10 s | 24 s + 17 s tmux setup | 17 s + 12 s |
| verifier | 2.6 s | 15.7 s | 18.6 s |
| wall outside Harbor's phases | ~0 s | 43 s (23 s of it client-side) | 67 s |
| episodes stopped by the 600 s agent timeout | 0 | 47% (1,764 of 3,757) | similar |
| forwarded tokens per step | 1.78M | 3.56M | 3.56M |

What's equal, so not a cause: the episode shape (calls, tokens per call, context), the concurrency actually reached (62 in flight in all three), staleness (TRL 1.5 mean against a cap of 4, so TRL never waits on it), the weight-sync pause (0.04 s), and vLLM itself. TRL's vLLM is underloaded, not slow: KV cache 2%, 8 requests running across 4 GPUs, no preemptions, prefix cache on with 75% hits, prefill 0.1 s per call, no top-k logprobs (`score_centering=False`). The 100% GPU utilization noted earlier is nvidia-smi's busy flag at batch ~2 per GPU, not saturation. The "model 242-265 s per episode" in the earlier notes is the collector's estimate (gap between agent steps minus requested waits), which counts Harbor-side overhead as model time. vLLM's own counters put real model time at 71 s.

## Root causes, ranked by seconds per step

Counterfactual chain from prime-rl's 187 s to TRL local's 601 s, using step = E x W / C. Each episode-wall delta is converted at prime-rl's E/C (1.01 s of step per 1 s of wall), and the E/C change comes last.

1. **The Harbor server's single event loop: +240 s/step** (238 s per episode in the agent loop). Each call takes 16 s from one agent step to the next, excluding the agent's requested wait. Only 3.1 s of that is vLLM, so ~13 s per call is spent waiting on the Harbor process. prime-rl's whole call cycle is ~7 s (4.5 s model + 2.4 s harness, including waits). Evidence that it's the shared loop and not the sandbox or vLLM:
   - The server's main thread used 456 of 527 CPU-minutes (87% of its lifetime) and sat at 93-99% in live samples. Its other threads were at ≤5%.
   - The HF and local runs show the same per-call gap (15.2 vs 16.1 s) on completely different sandbox backends.
   - Calls past the budget do no inference and run no command, yet still take 3.9 s each (120 s / 31 calls).
   - vLLM queue time is 0.2 ms, so nothing waits inside vLLM.

   Per call, the loop does a deepcopy plus JSON parse and serialize of a 20-50k-token chat in the proxy, parses `prompt_token_ids` and logprobs, does terminus-2's litellm call (including `collect_rollout_details=True`), and runs the tmux exec round trips. prime-rl spreads the same work over 64 processes, because terminus-2 runs inside each box, and it uses token-in/token-out `/inference/v1/generate`. I couldn't profile which of those dominates: py-spy needs to be the server's parent.
2. **Box lifecycle outside the agent: +87 s/step** (86 s per episode). TRL: 24 s box start, 17 s tmux setup, 16 s verifier and 43 s outside Harbor's phases, about 99 s in all. prime-rl: about 13 s. Part of the gap is real setup work: Harbor's singularity env bootstraps an in-box FastAPI server with apt-get and pip at every start. But the verifier takes 16 s against prime-rl's 2.6 s for the same `test.sh`, and 23 s of the outside-phase time is between the server finishing and the client seeing the result. That points at the same saturated loop.
3. **Episodes per step: +48 s/step.** TRL consumes 67.9 episodes per step against prime-rl's 62.6, because its step is 14 packed micro-batches x 4 ranks x 65,536 tokens and it refills zero-std groups, rather than "the next 8 groups". This is a config inequality (`GRAD_ACCUM=14`, sized from the probe), not a speed issue. The HF run consumed only 56.8 per step, which is why it averaged 496 s instead of 588 s.
4. **Budget tail: +34 s/step** (34 s per episode). The terminus-2 seam gets no `max_turns`, so the 30-call budget is enforced only by the capture proxy's canned stop reply. terminus-2 doesn't parse that reply as a command, so it re-asks until the 600 s agent timeout: 31 extra calls and 120 s on the 28% of episodes that hit the budget. These calls are also 29% of all requests the proxy handles (34.3k of 116.7k), which adds to the load in cause 1. prime-rl stops the episode at call 30.

The four add up to 187 + 240 + 87 + 48 + 34 = 596 s, against 601 s predicted and 588 s measured. The 600 s agent timeout hides how big cause 1 is: 47% of TRL episodes are cut at 600 s, prime-rl's never are (its longest stop is 30 calls at ~243 s). Without the timeout, TRL episodes and steps would run longer, and TRL episodes also get fewer turns before the cutoff than prime-rl's.

Not on the critical path today: fwd+bwd is 96 s vs 62 s. TRL forwards 3.56M tokens per step against 1.78M because terminus-2 re-renders the chat, so every turn forks (`rollout/fork_frac` 1.0) and tree packing shares only ~8x. prime-rl appends sampled ids, keeping one branch per episode (16x), and also prunes zero-advantage episodes before training. This becomes TRL's floor (~100 s per step) only once rollouts are 4x faster.

## Fixes

1. **Shard the Harbor server.** Run several `openenv harbor serve` processes, for example 8 servers x 8 sessions each, and spread sessions across them, so no single loop carries 64 agents. This is a launcher change plus a factory that round-robins `HARBOR_URL`s. It needs no TRL trainer change. Expected effect: the per-call cycle drops from ~17 s toward prime-rl's ~7 s, which removes most of causes 1 and 2, roughly 250-300 s per step. The right shard count needs one measurement (server CPU per shard and per-call gap). That means a GPU job, so I haven't run it. Longer term, OpenEnv's proxy should avoid the per-request deepcopy and JSON work on huge chats, or run it off the loop.
2. **Give terminus-2 the call budget.** Pass `max_turns=30` through the OpenEnv terminus-2 seam (it has no `step_limit` hook today, unlike mini-swe-agent's) so the agent stops at call 30, as in prime-rl. Saves ~34 s per step and 29% of proxy requests.
3. **Make the step take 8 groups.** Matching prime-rl's "next 8 groups, no refill" needs either a smaller `GRAD_ACCUM` (about 12-13 at the current ~1.34k samples per step) or a group-count-based step. Worth ~48 s per step at today's episode rate. This is a fairness fix for the comparison, not a TRL speedup.
4. **Box start (local variant only).** Prebake Harbor's in-box server into the `.sif` so the apt-get and pip bootstrap doesn't run at every start. This is a few seconds per episode.
5. Later, once rollouts are fast: training on appended tokens instead of re-rendered turns would halve forwarded tokens, but that's a harness/template property (`per_turn_capture_only`), not a TRL knob.

Config differences that make the two runs unequal, beyond the step-size gap in fix 3:
- prime-rl runs terminus-2 inside the box. SHARED_W2 says host-side.
- prime-rl's sticky router keeps an episode on one vLLM replica.
- prime-rl's summarisation can't be turned off (reactive only, 1.05 branches per episode).
- `max_staleness` is 8 in prime-rl and 4 in TRL. TRL never hits its cap, so this doesn't matter here.

The first two help prime-rl's speed. Neither explains a 3x gap on its own.
