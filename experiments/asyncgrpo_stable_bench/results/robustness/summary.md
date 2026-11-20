# Robustness summary

Generated 2026-11-14 07:46 UTC from `failures.jsonl` (157 rows) and `log_errors.csv`. The headline counts only `kind: failure` rows: the library itself broke, hung, crashed, gave wrong numbers, or its own docs, examples, defaults, lock or image didn't work as written. `failure_unattributed` (root cause unknown with our code in the failing path), `environment` (third-party stack problems any library on Qwen3.5 + B200 + torch 2.11-2.14 + cu130 would hit), `observation` (recovered warnings, cosmetic log lines, limitations not hit, API/design notes), `ours` (our harness and agents) and `infra` (spot preemption, node failures, cluster limits) are counted separately. Lib `all` holds problems that hit every library (cluster facts, shared tooling). Time lost is a rough estimate summed over `failure` rows that have one; `rows w/o time` counts rows without.

Scrutiny bias: TRL is the subject of the benchmark, so its logs and code were read far more closely than the others'. Observation counts are therefore not comparable across libraries; only the `failure` counts are meant for comparison.

## Headline per library

| lib | failures | blocker | major | minor | silent correctness | source patches | upstream-reportable | time lost (min) | rows w/o time | failure unattributed | environment | observation | ours | infra |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| trl | 2 | 0 | 2 | 0 | 0 | 0 | 2 | 0 | 2 | 0 | 4 | 13 | 6 | 1 |
| prime-rl | 6 | 0 | 3 | 3 | 0 | 1 | 4 | 81 | 0 | 0 | 1 | 7 | 4 | 1 |
| verl | 6 | 4 | 1 | 1 | 0 | 0 | 4 | 180 | 0 | 0 | 2 | 12 | 4 | 2 |
| skyrl | 12 | 4 | 6 | 2 | 3 | 2 | 12 | 173 | 6 | 0 | 2 | 5 | 5 | 1 |
| miles | 9 | 3 | 4 | 2 | 1 | 0 | 6 | 157 | 1 | 0 | 2 | 8 | 2 | 2 |
| openrlhf | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 30 | 0 | 0 | 1 | 6 | 5 | 1 |
| nemo-rl | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 20 | 0 | 0 | 0 | 0 | 1 | 0 |
| all | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 1 | 11 | 5 |

## Failures by category

| lib | install_break | missing_dep | docs_or_example_mismatch | unclear_setup | config_trap | crash | hang | silent_correctness | perf_trap | cluster_unfriendly |
|---|---|---|---|---|---|---|---|---|---|---|
| trl | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 |
| prime-rl | 1 | 0 | 0 | 0 | 1 | 3 | 0 | 0 | 0 | 1 |
| verl | 1 | 1 | 1 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| skyrl | 1 | 0 | 1 | 1 | 2 | 1 | 0 | 3 | 3 | 0 |
| miles | 0 | 0 | 1 | 2 | 2 | 0 | 0 | 1 | 3 | 0 |
| openrlhf | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| nemo-rl | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| all | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Excluded from the headline

| lib | ours rows | ours time (min) | infra rows | infra time (min) |
|---|---|---|---|---|
| trl | 6 | 140 | 1 | 21 |
| prime-rl | 4 | 25 | 1 | 15 |
| verl | 4 | 27 | 2 | 5 |
| skyrl | 5 | 210 | 1 | 6 |
| miles | 2 | 10 | 2 | 40 |
| openrlhf | 5 | 164 | 1 | 30 |
| nemo-rl | 1 | 25 | 0 | 0 |
| all | 11 | 115 | 5 | 35 |

## Log pattern totals for W1 runs

Raw line counts from `scan_logs.py` over every log file of the W1-S shared/recommended runs (runs still in progress are counted as of the scan). Warnings are dominated by a few repeated messages; see `log_messages_top.csv`.

| lib | runs | traceback | oom | nccl_error | timeout | retry/disconnect | cuda_error | nan_inf | killed/signal | error_other | warning_deprecation | warning_other |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| trl | 2 | 11 | 0 | 0 | 0 | 6 | 0 | 0 | 18 | 3 | 194 | 6908 |
| prime-rl | 2 | 37 | 0 | 0 | 0 | 0 | 0 | 0 | 31 | 10 | 248 | 32232 |
| verl | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| skyrl | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| miles | 1 | 5 | 0 | 0 | 0 | 30 | 0 | 0 | 0 | 4 | 191 | 78 |
| openrlhf | 3 | 0 | 22 | 0 | 0 | 0 | 0 | 0 | 2 | 45 | 100 | 137 |
| nemo-rl | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Blockers

| id | lib | kind | category | attributable | patched | symptom |
|---|---|---|---|---|---|---|
| verl-006 | verl | failure | install_break | lib | no | uv sync --frozen --extra vllm --extra fsdp on v0.9.1 fails: Failed to download flash-attn==2.8.3 ... 404. |
| verl-012 | verl | failure | crash | lib | no | Recipe jobs 364231 (step 1) and 364302 (step 6) crashed with KeyError: 'loss_mask' not found in TensorDict in compute_log_prob, each right after a validation pass. |
| skyrl-005 | skyrl | failure | silent_correctness | lib | no | With SkyRL's default remove_microbatch_padding=true on FSDP, the second sample of a packed micro-batch is off by up to 2.38 nats vs the sample alone (0.20-0.25 unpacked). No error or warning. |
| miles-006 | miles | failure | silent_correctness | upstream_dep (TransformerEngine 2.17.0+miles -> flash-attn-4 4.0.0b35 (shipped in the miles v0.1.3 image)) | no | With the recipe's --attention-backend flash on B200, trainer logprobs were -1.87/token vs SGLang -0.39 for the same tokens (forward max error 3.875 vs SDPA), and backward crashed: 'SM100 backward with head_dim=256 does not support local attention'. |
| openrlhf-004 | openrlhf | failure | silent_correctness | lib | no | With --ds.packing_samples, a 500-token sample packed after a 3000-token one differs from the sample alone by max /dlogp/ 7.94 (mean 2.24); at RL level vllm_kl was 0.18-0.25 packed vs 0.0003-0.0006 unpacked. No error or warning. |
| openrlhf-008 | openrlhf | ours | our_harness | ours | no | W1 run 364409 hung: step 48 generation stopped at 24/32 groups right after step 47's update; all 8 GPUs at 0% SM with memory held, every process idle, no log line for 30+ min. No checkpoint yet (first save at step 50), so 47 steps lost. |
| trl-011 | trl | ours | our_harness | ours | no | W1-S shared run 364684 FAILED at step 9/200, ~30 min after start: rank 0 raised 'RuntimeError: Rollout worker heartbeat stale: 300s > 300s; child is hung.' (async_rollout_worker.py:1627) while waiting on the sample queue; the other ranks died on a gloo 'Connection closed by peer'. Last logged step had rollout_queue_size 90 (22.8 one step earlier). |
| openrlhf-013 | openrlhf | ours | our_harness | ours | no | W1 rerun 364704 hung like 364409 (openrlhf-008): step 8 generation stopped at 31/32 groups after the trainer finished step 8 (rank 0 at 64/64 micro-steps); all GPUs idle, no error or traceback, cancelled after 29 min. |
| all-015 | all | environment | unclear_setup | upstream_dep (termgrade (ai-and/termgrade-environments)) | no | TermGrade task.toml files point at hb__<task>:latest, a local Docker tag; HF Sandboxes only pull prebuilt images from a registry, so no TermGrade task runs on HF Sandboxes as published. |
| all-016 | all | environment | install_break | upstream_dep (harbor 0.25.0 (hf_sandbox.py)) | yes | Every HF Sandbox upload (task environment/, verifier tests/) fails with tar 'Cannot change ownership', so no episode can be set up or verified. |
| all-017 | all | environment | silent_correctness | upstream_dep (termgrade (ai-and/termgrade-environments)) | no | Harbor's nop agent (does nothing) scores 1.0 on 15 of 128 candidate TermGrade tasks, so those tasks hand out free reward. |
| skyrl-012 | skyrl | failure | silent_correctness | lib | no | Every HF export of W1-S shared run 364970 has lm_head.weight != embed_tokens.weight (relative difference 2.4e-4 at step 50, 5.5e-4 at step 200) although Qwen3.5-2B has tie_word_embeddings=true; the trainer trains two matrices while vLLM keeps one shared matrix and skips lm_head.weight on load and sync. |
| skyrl-017 | skyrl | failure | config_trap | lib | yes | Stock HarborGenerator would zero the loss of almost every W2 group: an agent timeout masks its whole group of 8, and ~40-50% of 9B episodes hit the 600 s budget. |
| verl-023 | verl | failure | missing_dep | lib | no | As shipped: cannot train Qwen3.5-MoE with EP. verl's official image verlai/verl:vllm024.dev2 (the tag docker/Dockerfile.stable.vllm and the vllm CI workflow use) fails `import megatron.bridge` (No module named 'modelopt'), which verl's Megatron LoRA requires, and has no fla, which megatron-core's GatedDeltaNet requires. |
| miles-020 | miles | failure | perf_trap | upstream_dep (TransformerEngine 2.17.0+miles / flash-attn-4 4.0.0b35 (miles v0.1.3 image)) | no | W2 9B run 372222: episodes complete on local Apptainer boxes, but the first optimizer step OOMs in TE UnfusedDotProductAttention (softmax inp.float(): 'Tried to allocate 102.09 GiB', 84.7 GiB already in use, TP=4) on agentic samples up to the shared 65,536-token context. |
| verl-026 | verl | failure | crash | lib | no | verl v0.9.1 as shipped, FSDP2 actor + PEFT LoRA r32 (all-linear) with actor_rollout_ref.model.lora.merge=true, V1 separate_async, hybrid_engine=false, NCCL checkpoint engine, Qwen/Qwen3.5-35B-A3B, 4 vLLM 0.24 (TP1) + 4 trainer B200 (job 373603, text-to-SQL): the first weight sync into vLLM crashes in vllm qwen3_5.py load_weights -> load_fused_expert_weights -> routed_experts.weight_loader -> _get_hidden_dim: 'ValueError: shard_dim=0 is not a valid data dimension for a 3D tensor (expected 1 or 2)' (stdout.log line ~1218, 21:21:49Z, ~35 min after job start); the job then hangs with no further output. |
| nemo-rl-002 | nemo-rl | failure | crash | lib | no | NeMo-RL v0.8.0 NGC container, Automodel backend, built-in LoRA (policy.automodel_cfg.lora_cfg: r32/alpha 64, match_all_linear false, targets q/k/v/o + GDN in_proj_qkv/z/b/a + out_proj + shared_experts gate/up/down, 310 modules) on Qwen/Qwen3.5-35B-A3B with EP8 on one 8xB200 node (job 373868, otherwise the validated grpo-qwen3.5-35ba3b-dapo-4n8g-automodel-ep32 recipe): every policy worker fails at model setup in parallelize_model -> apply_fsdp -> fully_shard_by_dtype: 'ValueError: FSDP could not isolate parameters with a distinct dtype from siblings in the same module: linear_attn.out_proj.weight, linear_attn.in_proj_qkv.weight, linear_attn.in_proj_z.weight, linear_attn.in_proj_b.weight, linear_attn.in_proj_a.weight, mlp.shared_experts.up_proj.weight, ...' (train.log line ~276). Train exit at 01:10Z, before any rollout. |
| skyrl-025 | skyrl | failure | silent_correctness | lib | no | SkyRL W2-M with its own recommended agentic recipe (run_codecontest_fully_async.sh: rollout_is + geometric mask, stock HarborGenerator; run 374210, Qwen3.5-9B, local Apptainer) collapsed in 5 steps: train reward 0.203/0.086/0.016/0.047/0.000, agent timeouts 47->106 of 128, masked groups 12->16 of 16, grad norm 0 at step 5. No error raised. |
| miles-023 | miles | failure | perf_trap | lib | no | Audit rerun of miles v0.2.0 examples/text_to_sql (job 377684, same launcher as 373208 plus --sglang-moe-runner-backend triton, the option miles-022's assertion points at): SGLang ran the triton MoE runner, routed experts were captured, and the step-0 eval completed (validation 0.827, same as our base eval). The first train step then OOMed in Megatron attention (probs.bfloat16(), 'Tried to allocate 22.95 GiB', 162 GiB in use on GPU 7; 48.5 GiB on GPU 4). |

## Silent correctness

| id | lib | kind | category | attributable | patched | symptom |
|---|---|---|---|---|---|---|
| skyrl-005 | skyrl | failure | silent_correctness | lib | no | With SkyRL's default remove_microbatch_padding=true on FSDP, the second sample of a packed micro-batch is off by up to 2.38 nats vs the sample alone (0.20-0.25 unpacked). No error or warning. |
| skyrl-010 | skyrl | observation | silent_correctness | upstream_dep (flash-linear-attention 0.5.2 + triton 3.7.1 (triton-lang/triton#10590)) | no | Latest release skyrl-v0.4.0 ships an unpinned flash-linear-attention; SkyRL's own lock comment says fla 0.5.2 + Triton 3.7.1 corrupts GDN outputs in ~40% of calls on B200. |
| miles-006 | miles | failure | silent_correctness | upstream_dep (TransformerEngine 2.17.0+miles -> flash-attn-4 4.0.0b35 (shipped in the miles v0.1.3 image)) | no | With the recipe's --attention-backend flash on B200, trainer logprobs were -1.87/token vs SGLang -0.39 for the same tokens (forward max error 3.875 vs SDPA), and backward crashed: 'SM100 backward with head_dim=256 does not support local attention'. |
| openrlhf-004 | openrlhf | failure | silent_correctness | lib | no | With --ds.packing_samples, a 500-token sample packed after a 3000-token one differs from the sample alone by max /dlogp/ 7.94 (mean 2.24); at RL level vllm_kl was 0.18-0.25 packed vs 0.0003-0.0006 unpacked. No error or warning. |
| all-017 | all | environment | silent_correctness | upstream_dep (termgrade (ai-and/termgrade-environments)) | no | Harbor's nop agent (does nothing) scores 1.0 on 15 of 128 candidate TermGrade tasks, so those tasks hand out free reward. |
| skyrl-012 | skyrl | failure | silent_correctness | lib | no | Every HF export of W1-S shared run 364970 has lm_head.weight != embed_tokens.weight (relative difference 2.4e-4 at step 50, 5.5e-4 at step 200) although Qwen3.5-2B has tie_word_embeddings=true; the trainer trains two matrices while vLLM keeps one shared matrix and skips lm_head.weight on load and sync. |
| miles-018 | miles | observation | silent_correctness | lib | no | miles' own Qwen3.5-35B-A3B LoRA CI is pinned to 8x H200 because on 8x B200 its first combo fails the step-0 check_kl (ppo_kl 1.6e-4 against the 1e-8 LoRA limit; H200 logs 0.0). |
| verl-021 | verl | observation | silent_correctness | lib | no | With LoRA in V1 separate_async and the default model.lora.merge=False, the weight sync streams only base weights and skips the lora_ keys, so vLLM would keep sampling the frozen base model with no error (found by reading code, not run). |
| skyrl-025 | skyrl | failure | silent_correctness | lib | no | SkyRL W2-M with its own recommended agentic recipe (run_codecontest_fully_async.sh: rollout_is + geometric mask, stock HarborGenerator; run 374210, Qwen3.5-9B, local Apptainer) collapsed in 5 steps: train reward 0.203/0.086/0.016/0.047/0.000, agent timeouts 47->106 of 128, masked groups 12->16 of 16, grad norm 0 at step 5. No error raised. |
| trl-033 | trl | observation | silent_correctness | lib | no | Text-to-SQL 35B stock run 373132 trailed prime-rl and SkyRL, mostly on hard (0.626 vs 0.707 at step 80 after the fix); no error (#8212). |

## Every row by kind

Why each row got its `kind` (`retag_note` in the ledger).

| id | kind | severity | note |
|---|---|---|---|
| prime-rl-001 | failure | minor | README Manual Setup doesn't work as written (SSH submodule URLs, git-lfs). |
| prime-rl-005 | failure | minor | Importing prime_rl breaks transformers' Qwen3.5 classes in the same process (global side effect); hit in our check only. |
| prime-rl-007 | failure | major | Fixed default ports crashed a real smoke on a shared node. |
| verl-001 | failure | minor | Clone instructions leave recipe/ empty (submodule not initialised). |
| verl-006 | failure | blocker | verl's own v0.9.1 lock pins a 404 wheel URL (lock exception to environment). |
| verl-009 | failure | major | Documented hybrid_engine=False path crashes at init in v0.9.1; source patched. Audit retag 2026-11-14: blocker -> major: verl's own VERL_LOGGING_LEVEL=WARN runs this path (used for W1 run 370023) (audit M3). |
| verl-012 | failure | blocker | TransferQueue drops data after ACK timeouts, then KeyError crash. |
| skyrl-005 | failure | blocker | Default packing silently corrupts Qwen3.5 logprobs (cf. openrlhf-004). |
| skyrl-006 | failure | major | Default num_workers=8 with spawn contradicts its own docstring and stalled eval 7+ min. |
| skyrl-009 | failure | major | Shipped recipe script no longer matches the documented published run (inferred from #2635, not run). |
| miles-006 | failure | blocker | miles' own shipped image + recipe backend gives wrong logprobs on B200 (image exception to environment); runtime patch. |
| miles-008 | failure | minor | Documented flag asserts at startup without an undocumented companion flag. |
| miles-009 | failure | major | Shipped example script fails to start with the v0.1.3 checkpoint layout. |
| miles-010 | failure | major | --save-hf exports an incomplete checkpoint that vLLM/transformers can't load as the original model. |
| openrlhf-004 | failure | blocker | Packing silently corrupts Qwen3.5 logprobs (cf. skyrl-005). |
| prime-rl-010 | failure | major | The shipped converter crashed on every rank with documented usage; source patched. |
| trl-013 | failure | major | Latent bug in TRL's own reward helper, found while root-causing trl-011; the mechanism is reproduced with TRL's rollout worker, the trigger is a model output. |
| trl-014 | failure | major | A TRL default (score centering) silently costs ~1/3 of throughput through the vLLM endpoint it uses; measured on the live recommended run and A/B'd against the shared arm. |
| skyrl-011 | failure | major | One transient non-JSON server response kills a whole SkyRL fully-async run; no harness code in that path. |
| skyrl-012 | failure | blocker | SkyRL's own FSDP init silently trains a different model than vLLM samples for tied-embedding checkpoints. |
| skyrl-014 | failure | major | The library's own lock can't run the sandbox backend the workload needs. |
| skyrl-016 | failure | minor | Needed reading the entrypoint source to configure a non-Daytona backend. |
| skyrl-017 | failure | blocker | Library default silently discards most agentic training signal under a wall-clock budget. |
| skyrl-018 | failure | major | Silent compute waste in the library's padded path; affects the W1 runs. |
| skyrl-019 | failure | minor | Registry API has no way to opt out of the extra forward. |
| prime-rl-012 | failure | major | Shipped docker runtime breaks on rootless Docker/podman as written; workaround is outside the library. |
| miles-016 | failure | major | Consequence of miles' own shipped image (same exception as miles-006): the shipped correct path is much slower. |
| verl-023 | failure | blocker | verl's own recommended image can't train this model on its Megatron/EP path. |
| miles-019 | failure | major | Shipped async agentic recipe fails miles' own argument validation. |
| prime-rl-018 | failure | minor | LoRA export gap in prime-rl's own tooling, worked around with bench tooling only. |
| miles-020 | failure | blocker | Consequence of the B200 attention-backend bug at agentic sequence lengths. |
| miles-022 | failure | minor | Shipped text-to-SQL MoE recipe (routing replay) is not runnable on B200 with the default SGLang MoE runner. Audit retag 2026-11-14: blocker -> minor config_trap: audit rerun 377684 with the shipped --sglang-moe-runner-backend triton (the runner the assertion names) gets past this to the step-0 eval; the run then OOMs (miles-023) (audit M5). |
| verl-026 | failure | blocker | verl's merged-LoRA sync sends fused 3D expert tensors that vLLM's Qwen3.5-MoE loader rejects; no built-in option avoids it. |
| nemo-rl-002 | failure | blocker | Built-in Automodel LoRA cannot set up Qwen3.5-35B-A3B under EP+FSDP; LoRA comparison with TRL not possible on NeMo-RL as shipped. |
| skyrl-024 | failure | major | config_trap: SkyRL default failure handling, no harness change; trains on a quarter of each batch. |
| skyrl-025 | failure | blocker | failure: SkyRL recipe as shipped, only workload-forced deviations; collapse with no error. Audit retag 2026-11-14: stands. Rerun 377353 with max_concurrency=64 (the only change) also produced no step 1 in 90 min (83 agent_timeout of ~109 trials), so it isn't our concurrency setting; cause of the slow episodes unknown (audit F2). |
| miles-023 | failure | blocker | Audit rerun: miles-022 is avoidable with a shipped flag; with it, miles still can't train the 35B text-to-SQL recipe on one B200 node, now from unfused-attention OOM (cf. miles-020). |
| trl-002 | environment | minor | uv keys --no-build-isolation builds on version, not torch ABI; hits any lib on this stack (cf. verl-007). |
| trl-003 | environment | major | No causal-conv1d wheel for torch 2.14 + cu130 and no nvcc/g++ on CPU nodes; TRL pins no causal-conv1d, same as verl/OpenRLHF. |
| trl-005 | environment | minor | vLLM's PyPI wheel lacks FlashInfer precompiled kernels; any lib serving with vLLM 0.33 from PyPI hits it. |
| trl-006 | environment | minor | flashinfer-cubin has no 0.7.x on PyPI; a FlashInfer packaging gap, not TRL's lock. |
| prime-rl-002 | environment | minor | mamba-ssm silently skips its CUDA extension without a toolkit; any lib building it on a CPU node hits it. Not needed for W1. |
| verl-007 | environment | minor | uv cache reuse across torch ABIs (cf. trl-002). |
| verl-008 | environment | major | No causal-conv1d wheel for torch 2.11 and no toolchain on CPU nodes (cf. trl-003). |
| skyrl-002 | environment | major | math-verify antlr4 extra vs omegaconf's antlr4 pin; any omegaconf-based lib using math-verify hits it. |
| miles-007 | environment | minor | TE/cuDNN has no sm100 head_dim-256 backward; an alternative backend we tried, not the recipe default. |
| openrlhf-002 | environment | major | No flash-attn/causal-conv1d wheels for torch 2.13 + cu130 and no CPU-node toolchain (cf. trl-003, verl-008). |
| all-001 | environment | major | nvcc/g++ missing on CPU nodes; hits every lib that builds CUDA extensions. |
| all-015 | environment | blocker | Dataset/backend mismatch every library running TermGrade on HF Sandboxes hits; not any RL library's code. |
| all-016 | environment | blocker | Harbor bug on the shared sandbox backend; any library running Harbor + HF Sandboxes hits it. |
| all-017 | environment | blocker | Dataset defect that would inflate every library's reward equally; found by our nop validation. |
| skyrl-015 | environment | major | Harbor bug any library on hf-sandbox hits; already known from TRL. |
| all-020 | environment | minor | Harbor default on this Apptainer; hits any library using Harbor's singularity env. |
| miles-021 | environment | major | Local sandboxes for miles' Harbor path need nested Apptainer workarounds; miles targets cloud sandboxes. |
| trl-007 | observation | info | Retries recovered every request; nothing broke. |
| trl-008 | observation | info | Cosmetic vLLM warning; tied head verified updated (IS ratio 1.000). |
| trl-009 | observation | info | API/design note: no config switch for these behaviours; nothing broke. |
| trl-010 | observation | info | Limitation of adapter sync with vLLM DP>1; not hit (W1 is full fine-tuning). |
| prime-rl-006 | observation | info | Deliberate empty-run-dir guard working as designed; our launcher created the dir. |
| prime-rl-008 | observation | info | Platform-oriented defaults (Prime judge, monitor) turned off up front; not hit. |
| verl-003 | observation | info | Design note: the data prep script imports verl, so it needs the full venv; nothing broke once run in the venv. |
| verl-005 | observation | info | Config field with no effect on the FSDP path; harmless with text-only data, not hit. |
| verl-013 | observation | info | Hydra config inheritance limit; worked around with a standalone config, nothing broke. |
| verl-014 | observation | info | Fixed ports noted, not hit (exclusive nodes). |
| verl-015 | observation | info | API note: no off-switch for dual-clip; nothing broke. |
| skyrl-007 | observation | info | Repeated warning from an optional monitor without a dashboard; nothing broke. |
| skyrl-008 | observation | info | Correct constraint on our invalid config; the late check cost a job, a design note. |
| skyrl-010 | observation | info | Known fla+Triton issue documented in SkyRL's own lock comment on the release; we used main, not hit. |
| miles-001 | observation | info | Launchers' pkill/ray stop assume a dedicated node; bypassed up front, never run, so not hit. |
| miles-002 | observation | info | No shipped model args for the 2B size; not a break. |
| miles-011 | observation | info | Default leaves staleness uncapped; recorded design note, nothing broke. |
| miles-012 | observation | info | Documented deadlock risk of a default; not hit. |
| miles-013 | observation | info | Published recipe layout is slow at this length but ran; perf note. |
| openrlhf-005 | observation | info | API note: no chat_template_kwargs; prompts pre-rendered instead. |
| openrlhf-007 | observation | info | enforce_eager suits the colocated examples it ships in; we copied it into a non-colocated config. |
| openrlhf-009 | observation | info | Metric semantics note: timing/broadcast includes lock wait; numbers correct for what is measured. |
| openrlhf-010 | observation | info | API note: no max-steps flag. |
| openrlhf-012 | observation | info | Allocator retries recovered; impact not measured. |
| trl-012 | observation | info | dtype-cast warning; measured sync ~0.2 s, no measured impact. |
| openrlhf-014 | observation | info | Recovered allocator retries; supersedes openrlhf-012's open question (not related to the hang). |
| trl-019 | observation | info | Harness/template property every library sees; matters for throughput, not correctness. |
| trl-020 | observation | info | Rare recovered failure in an upstream dependency; no lost run. |
| prime-rl-013 | observation | info | No switch for summarisation; did not break a run. |
| prime-rl-014 | observation | info | Config-side workaround, nothing broke. |
| miles-015 | observation | info | Design limit, nothing broke yet. |
| verl-018 | observation | info | Missing feature, not a breakage; it blocks W2 on verl. |
| prime-rl-015 | observation | info | Documented semantics, nothing broke; noted because it changes episode length if picked. |
| trl-021 | observation | info | No knob for env kwargs, nothing broke; an env var covered it. |
| skyrl-021 | observation | info | Costs disk and save time, nothing broke. |
| skyrl-022 | observation | info | Duplicate save, nothing broke. |
| trl-022 | observation | info | Handled by TRL as designed (unscorable, excluded); cost ~2% of episodes; OpenEnv-side cause. |
| prime-rl-017 | observation | info | Hazard for any lib relying on the template default at M; nothing broke here. |
| verl-019 | observation | info | Workaround for verl-009 found by reading source; the crash itself stays counted as verl-009. |
| all-022 | observation | info | Hazard that didn't break anything; applies to any library on Harbor's singularity env. |
| miles-017 | observation | info | Documented limitation found by reading the shipped launcher; not hit in a run. |
| miles-018 | observation | info | Upstream-documented hazard on our hardware, not hit by us. |
| verl-020 | observation | info | Design note (no EP switch on FSDP2); not a run failure. |
| verl-021 | observation | info | Silent-correctness default found by code reading; we never ran it, so it isn't counted as a failure. |
| verl-022 | observation | info | verl's own v0.9.1 lock pins 404 wheel URLs (lock exception to environment, as verl-006). Audit retag 2026-11-14: duplicate of verl-006 (same root cause, other install extra); kept for the record, counted once under verl-006 (audit M3). |
| verl-024 | observation | info | verl's own recommended extra can't train this model on its Megatron/EP path. Audit retag 2026-11-14: duplicate of verl-023 (same root cause, other install extra); kept for the record, counted once under verl-023 (audit M3). |
| verl-025 | observation | info | Observation: the library ships an extension API but no environment; the glue is ours and is counted as workload integration. |
| prime-rl-019 | observation | info | Built-in verifiers budget behaviour; more visible at 35B because episodes run longer. |
| trl-030 | observation | info | Found by the benchmark, fixed in TRL v1.18.0, so not counted against 1.18 (audit F3). |
| trl-031 | observation | info | Found by the benchmark, fixed in TRL v1.18.0, so not counted against 1.18 (audit F3). |
| trl-032 | observation | info | Found by the benchmark, fixed in TRL v1.18.0, so not counted against 1.18 (audit F3). |
| trl-033 | observation | info | Found by the benchmark, fixed in TRL v1.18.0, so not counted against 1.18 (audit F3). |
| trl-004 | ours | minor | Our kernel-binding probe. |
| prime-rl-004 | ours | minor | Our Qwen3.5 check. |
| prime-rl-009 | ours | minor | Our agent loop. |
| verl-004 | ours | minor | Our hf CLI usage. |
| verl-010 | ours | minor | Our check's use_cache default. |
| verl-011 | ours | major | Our Ray port range. |
| verl-017 | ours | minor | Our chain design. |
| skyrl-003 | ours | major | Our Ray launcher. |
| skyrl-004 | ours | minor | Our Ray head address. |
| miles-004 | ours | minor | Our non-atomic image import. |
| miles-014 | ours | minor | Our eval symlinks. |
| openrlhf-003 | ours | minor | Our agent's permission classifier. |
| openrlhf-006 | ours | minor | Our GCS port choice. |
| openrlhf-008 | ours | blocker | Same mechanism as openrlhf-013: our shared reward (math_verify off the main thread, timeouts disabled) can freeze the rollout actor via a GIL-held sympy call; not a captured trace, so unattributed. The same mechanism was reproduced end to end for TRL's worker (trl-011). Audit retag 2026-11-14: ours, same as openrlhf-013 and trl-011: our shared math reward (math_verify without timeouts off the main thread) froze the rollout actor (audit m3). |
| openrlhf-011 | ours | minor | Our time limit. |
| all-002 | ours | minor | Our resource asks. |
| all-003 | ours | major | Our queue env leak. |
| all-004 | ours | major | Our waiter requeue. |
| all-005 | ours | major | Our waiter naming. |
| all-006 | ours | major | Our node guard/FIFO. |
| all-007 | ours | major | Our shared config. |
| all-008 | ours | minor | Our token counting. |
| all-009 | ours | minor | Our eval typo. |
| all-010 | ours | minor | Agent permissions. |
| all-011 | ours | minor | Our stall watchdog. |
| all-012 | ours | minor | Our collectors. |
| prime-rl-011 | ours | major | Our queue chaining. |
| trl-011 | ours | blocker | Root-caused to our shared reward, not TRL; the TRL watchdog fired correctly. |
| openrlhf-013 | ours | blocker | Our shared reward runs in the frozen path, triggered by OpenRLHF running custom rewards in-thread with no timeout; covers openrlhf-008 too (same mechanism), which should move to failure_unattributed. Audit retag 2026-11-14: ours, same cause as trl-011: our shared math reward holding the GIL in sympy (audit m3). |
| trl-015 | ours | minor | Our patch, our mistake. |
| trl-016 | ours | minor | Our launcher; the hang-after-failure is vLLM's but only cost time because our check relied on the process exiting. |
| trl-017 | ours | minor | Our sweep script. |
| trl-018 | ours | minor | Our launcher ordering. |
| skyrl-013 | ours | major | User decision 2026-11-10: the shared arm only uses what each library ships. SkyRL's fully async mode refuses PPO-clip ('regular'); its built-in rollout_is with the shared 0.2/0.28 band masks both sides and collapsed the run. A one-sided custom loss (ppo_rollout, envs/skyrl/collapse_ppo_rollout_main.py) stayed stable in a 40-step check but is not used, since it isn't part of SkyRL. Audit retag 2026-11-14: our harness/config. We mapped PPO's 0.2/0.28 eps onto rollout_is's two-sided mask band; SkyRL's own async recipe uses (0.5, 5.0) plus a geometric sequence mask, and SkyRL's default dppo loss on the same W1 config (run 365171) trained cleanly to 0.410/0.329/0.916 at step 200 (audit F1). |
| skyrl-020 | ours | major | Shared harness script. |
| prime-rl-016 | ours | minor | Our export helper, caught before any run. |
| skyrl-023 | ours | major | failure: SkyRL's built-in loss, no harness change; finished with exit 0 but the eval falls below the base model. Audit retag 2026-11-14: our harness/config. Same mapping as skyrl-013 (shared 0.2/0.28 band on rollout_is) at 9B; no 9B run with SkyRL's default dppo loss exists (audit F1). This is the W1-M math run, not W2. |
| nemo-rl-001 | ours | major | Validated Qwen3.5-35B-A3B recipe OOMs on the first train step at 29.7k-token sequences on one node; config-level fix available. Audit retag 2026-11-14: our deviation. The recipe is validated on 4 nodes at 9,216-token sequences; we ran it on 1 node at 29,696 tokens (~3x) without the shipped train_mb_tokens / expandable_segments knobs (audit M2). |
| trl-001 | infra | minor | Spot preemption of the install job. |
| prime-rl-003 | infra | minor | Spot preemption plus slow /fsx imports. |
| verl-002 | infra | minor | Spot preemption. |
| verl-016 | infra | minor | Spot NODE_FAIL. |
| skyrl-001 | infra | minor | Spot preemption. |
| miles-003 | infra | major | Spot preemption. |
| miles-005 | infra | minor | Slow small-file checkout on /fsx. |
| openrlhf-001 | infra | major | Spot preemption. |
| all-013 | infra | minor | ptrace restricted on compute nodes. |
| all-014 | infra | minor | Cold imports from /fsx on the cluster. |
| all-018 | infra | minor | Cluster network fact. |
| all-019 | infra | major | Node OS package version. |
| all-021 | infra | minor | Old podman on the GPU nodes under concurrent starts; hits any library on local podman. |
