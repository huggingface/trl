# Harbor/Pi swarm training

Train a fixed team of Pi agents with `AsyncGRPOTrainer`. Each agent runs inside its own
native Harbor task environment. Teammates share the same **huggingface/msgboard** used
by Flock, while their AgentEscape rooms remain independent.

One rollout is one team attempt, not one agent. Independent attempts at the same task
form a GRPO group. Every agent receives its team's advantage. The loss averages each
team's supervised tokens first, then averages teams. Variable-length trajectories and
Pi history forks therefore do not give a team extra weight.

This combines the team-rollout approach in [TRL #8424](https://github.com/huggingface/trl/pull/8424)
with the Pi capture approach in the
[agent-swarm experiment](https://github.com/huggingface/trl-internal/tree/prj-agent-swarm-train/experiments/agent_swarm_train).
Unlike that experiment, all agents train, with no frozen workers or coordinator.

## Install and runtime

Use Python 3.12+, the TRL checkout, a working Docker or rootless Podman installation,
and separate GPUs for inference and training. The default is single-rank LoRA;
full language-model fine-tuning supports FSDP2 and a call-boundary policy gateway.
Both paths use full-vocabulary sampling. The runtime does not depend on Flock.

```sh
uv pip install -e '.[dev]' -r examples/async_grpo_harbor_swarm/requirements.txt
```

Run vLLM in a separate environment or container with a compatible release from TRL's
`vllm` extra. The two processes must see the trainer output directory at the same
absolute path: vLLM loads versioned adapters from it. Choose a CUDA build compatible
with your GPU driver. Installing this example does not install vLLM.

On Blackwell trainer GPUs, install `nvidia-cutlass-dsl==4.8.1` in the trainer environment
for the FlashAttention-4 kernel selected by AsyncGRPO. The Slurm launcher resolves
the selected attention kernel before collecting native rollouts, so missing kernel
dependencies fail before an expensive batch.

The task repository is private; use an existing HF login or `HF_TOKEN` with read access.
Tasks are downloaded as real files into the run directory, pinned to
`144b647af561e28d681552cb04bc7b4a8dbeb1ce`. There is no task rewrite or shared room.
These multi-container tasks require Docker/Podman, not HF Sandbox.

Cluster Podman installations without a user systemd session need their existing native
health-check scheduler and job-local container storage. Establish that setup before
launching; the example does not change machine-wide container settings. Start with the
native Harbor oracle on a downloaded task to check the runtime and verifier.

Use Podman 4.9 or newer. With the pinned `podman-compose` 1.x, put the included Harbor
compatibility wrapper on `PATH` (activate `.venv` first):

```sh
compose_bin=$(mktemp -d)
ln -s "$PWD/examples/async_grpo_harbor_swarm/podman_compose_compat.py" "$compose_bin/podman-compose"
export PATH="$compose_bin:$PATH"
export PODMAN_COMPOSE_PROVIDER="$compose_bin/podman-compose"
```

It translates Harbor's `exec`/`cp` commands and declares Compose's implicit default
network through a runtime overlay. Docker Compose does not need this wrapper.

## Training smoke

Start a fresh policy server (example command from the vLLM environment):

```sh
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 \
vllm serve Qwen/Qwen3.5-4B --host 0.0.0.0 --port 8000 \
  --max-model-len 16384 --max-num-seqs 8 --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --language-model-only \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --logprobs-mode processed_logprobs --return-tokens-as-token-ids \
  --enable-lora --max-lora-rank 32 --max-loras 6
```

Set `CAPTURE_URL` to the trainer host's address reachable **from the task containers**,
for example `http://10.0.0.10:8300`. A container's `localhost` is not the host. The capture
service listens on port 8300 for the life of the runner. No public tunnel is required.
If the policy server requires authentication, set `VLLM_API_KEY` in the trainer environment.

```sh
CUDA_VISIBLE_DEVICES=1 python examples/async_grpo_harbor_swarm/async_grpo_harbor_swarm.py train \
  --vllm-url http://127.0.0.1:8000 --capture-url "$CAPTURE_URL" \
  --backend podman --num-tasks 4 --max-steps 2 --max-groups 32 \
  --output-dir runs/swarm-smoke
```

Defaults: Qwen3.5-4B, LoRA rank 32/alpha 64, learning rate 1e-5, two agents per team,
four team attempts per GRPO group, four complete teams per update, two concurrent teams,
temperature 1, 16 calls/agent, 1,024 output tokens/call, and 300 seconds/agent. The task's
native 500-action budget remains in force. `--help` lists the adjustable limits.

Difficulty-5 tasks are sorted and shuffled with seed 42: 48 train, 12 held out.
`--num-tasks` selects a prefix of the corresponding split; it does not change the split.
Use a fresh output directory for each run. Resume is not implemented.

For a longer run, increase `--max-steps` and use `--time-limit-seconds` to stop and save
at the next optimizer step after the training-time budget. Keep a scheduler wall-time
limit as well, since environment setup and a pending rollout can outlast that budget.
Add `--report-to trackio --trackio-space-id YOUR_ACCOUNT/YOUR_SPACE --run-name RUN_NAME`
to log training metrics to a private Trackio Space. `--trackio-project` defaults to
`trl-harbor-swarm`. Set `TRACKIO_DIR` to node-local storage; SQLite initialization hung
on the shared cluster filesystem during validation. Keep the local logs with run artifacts
after shutdown; remote metrics are synced to the Space's bucket.

A smoke only demonstrates a learning update if a naturally mixed-reward group was
observed and trainable parameters changed. Equal-reward groups are retained with zero advantage. A completed run with no
mixed group saves its checkpoint and reports that the learning criterion remains unmet.
`--max-groups` also bounds collection when environments repeatedly fail; zero disables
that collection limit. Native training uses only verifier rewards. The separate
`--synthetic-smoke` engineering check uses explicitly synthetic rewards to test gradients
and weight transfer; its outputs are not benchmark results.

## Full-model training

`--full-finetune --gateway-url URL` selects full language-model updates, with the vision
tower frozen. Run `policy_gateway.py` in front of a fresh vLLM server configured with
`--weight-transfer-config '{"backend":"nccl"}'` and no LoRA. Set the same private
`SWARM_GATEWAY_TOKEN` in the gateway and trainer. Install a matching vLLM weight-transfer
client in the trainer environment as well as the server. Validate imports on a GPU;
the inference engine and trainer have different dependency requirements.

During synchronization the gateway blocks new calls and drains active ones. TRL transfers
weights, resets the prefix cache, and publishes the new version before calls resume.
Teams may span versions. Exact prompt/output token fingerprints join gateway provenance
to capture nodes; missing or ambiguous matches make a team unscorable. Behavior log-probs
and masks remain the ones captured during sampling. Control failure closes admission.

For the larger experiment, the configuration is:

- 16 agents/team; 16 teams/GRPO group; 64 teams/update (1,024 agent trajectories).
- 48 training and 12 held-out scenarios **per difficulty**, selected with
  `--difficulties 10 15`; the two difficulty pools are interleaved.
- 1,800 seconds and 64 calls/agent; maximum staleness 4; fresh Qwen3.5-4B weights.
- Four FSDP2 ranks, each accumulating sixteen complete teams per optimizer update.
  Ragged chunks are padded with zero-gradient work so ranks execute matching collectives.
- Target 16 concurrent teams (256 agents), with optional validation at 4, 8, and 16 teams.

`full_run.sbatch`, `full_server.sh`, and `full_node.sh` provide the cluster launcher.
Set `SWARM_REPO`, `SWARM_RUN_DIR` (fresh), `SWARM_MODEL_SNAPSHOT`, and
`SWARM_PODMAN_BUNDLE`. The launcher uses one eight-H100 node for eight hours, including
validation: four GPUs and 64 CPUs for training/Harbor, and four GPUs and 24 CPUs
for inference. Exclusive Slurm steps reserve disjoint GPU sets, with 1,400 GiB and
512 GiB memory limits respectively. Its cluster-specific partition and container-image paths must match the
installation. By default, it starts native training directly from base weights with 16 concurrent teams.
Set `SWARM_VALIDATE=1` to opt into two synthetic-reward updates, checkpoint reload,
and native concurrency checks before training. With validation enabled, the server
restarts after the synthetic smoke so real training still starts from base weights.

For AgentEscape, build the reusable Pi 1.4.0 image once using the same pinned task snapshot:

```sh
python examples/async_grpo_harbor_swarm/prepare_agent_image.py \
  --task-dir /path/to/tasks/agentescape-10-skeleton-TASK_ID --output-dir /path/to/agent-image
export SWARM_AGENT_IMAGE_ARCHIVE=/path/to/agent-image/agent-image.tar
export SWARM_AGENT_OVERLAY=/path/to/agent-image/agent-overlay.json
```

The launcher loads this image before starting rollouts. It contains Node, Pi, and the
existing Flock board client; puzzle and verifier services still use the task's own code.
The Podman backend limits environment starts/stops to eight at a time and serializes
builds because AgentEscape tasks share image tags. Daemon detection runs off the event
loop and is cached. Compose cancellation terminates its child processes. Healthcheck
discovery retries timeouts, caches container configuration, and limits concurrent checks
to sixteen. First readiness checks take priority over periodic checks of containers
that have already passed. A healthcheck scheduler or gateway exit fails the training job promptly.
The job-local container configuration disables per-container kernel keyrings; these
tasks do not use them, and the default per-user quota prevents the 256-agent target
from starting all of its containers.
At most two teams enter setup together; admission happens before trial creation and
startup timers. Once all members are ready, the next team can enter setup while the
ready team executes. This preserves the target of sixteen executing teams without
starting hundreds of trial timers while their environments are still queued.
After agent execution, at most eight trials collect artifacts, run native grading,
and clean up together. Admission uses Harbor's agent-end hook, outside the agent's
execution budget and before the verifier's timeout starts.

`--teams-per-update` always counts global teams; it must divide evenly across ranks.
The loader stages a complete optimizer batch before yielding any accumulation rounds. While waiting for rollouts, all ranks exchange queue status every five seconds instead of leaving peer ranks in one long collective. Collection errors and the final partial batch are reported to every rank. `--distributed-timeout` controls both the trainer and team-transfer process groups.
`--admission-deadline` and `--collection-deadline` are absolute Unix timestamps. Admission
closes 45 minutes before allocation expiry; collection stops ten minutes before expiry,
leaving time to save. Partial final batches are discarded. Checkpoints retain full model
and optimizer state, with the latest three kept. Resume remains unsupported.

Use `monitor_full.py --run-dir DIR --job JOB_ID` to monitor queued jobs, validation stages,
training rewards by difficulty, active teams, and saved steps in the existing private
Trackio Space. Training metrics also include staleness, queue depth, memory, and sync time.
Native gate results and synthetic smoke results remain separate from training metrics.

## Evaluation and artifacts

To verify the saved adapter, register `runs/swarm-smoke/checkpoint` as a named LoRA on
the same base-model vLLM server (using its `/v1/load_lora_adapter` endpoint), then run:

```sh
python examples/async_grpo_harbor_swarm/async_grpo_harbor_swarm.py evaluate \
  --vllm-url http://127.0.0.1:8000 --capture-url "$CAPTURE_URL" \
  --served-model swarm-checkpoint --num-tasks 2 --output-dir runs/swarm-eval
```

For a baseline, omit `--served-model`. Evaluation uses the same Pi runtime, communication,
budgets, capture validation, and native verification as training.

- `manifest.json`: task revision, exact split, selected tasks, settings, dependency versions.
- `teams.jsonl`, `groups.jsonl`: team outcomes and training group reward variation.
- `teams/<id>/`: per-agent captures/transcripts, Harbor trials, native verifier evidence,
  trusted success evidence, and board `messages.jsonl`.
- `checkpoint/`: saved adapter and tokenizer; `training-smoke.json` records completed
  optimizer steps and mixed groups. `evaluation.json` reports scored and unscorable teams separately.

Raw Harbor trial configs and msgboard databases are local runtime state, not shareable
artifacts. Export only the sanitized team/capture/message files; never publish tokens or
the operator-token file. Agent tokens are revoked when their team's board closes.

## Execution and loss details

Pi's shell uses `flock-tool message_board.*`, with Flock's unchanged SDK operations.
Each attempt has a separate native msgboard, `general` channel, authenticated identities,
and mounted Unix socket. This is Flock's basic local-board mode, without its optional
rate-rule proxy, controller, or Space deployment. A 500-character post limit is prompt
guidance, not a new board policy.

Agents wait at a readiness barrier after setup. A host-side watcher checks the trusted
puzzle service for `won` and the pinned scenario hash. On first success it asks Pi to
abort through a native extension, then lets Harbor collect evidence and grade. Reward
is 1 only for native verified success; a valid unsuccessful attempt receives 0. Infrastructure,
capture, or verifier failures are unscorable. Incomplete model responses are never reconstructed.
Trusted-state reads and Pi stop signals allow 30 seconds per container command and retry
a timeout once. Both operations are idempotent. Persistent timeouts still make the team
unscorable; errors identify the team, agent, and operation. Other errors and cancellation
are not retried.

The capture proxy supplies token IDs, logprobs, and explicit loss masks. In LoRA mode, each team pins a
named policy adapter. Full-model mode records each call's serving version and drops a whole
team if its oldest call is too stale. `SwarmTrainer` consumes a complete team as one logical micro-batch,
splits its rows into bounded forward/backward chunks, and uses the team's total supervised
token count for every chunk. The parent trainer supplies optimizer steps, checkpointing,
and policy synchronization. Staleness filtering discards complete teams.

The policy objective follows the s3 example: shared standardized team advantages,
importance-ratio mask [0.2, 5.0], effectively disabled PPO clipping, and no KL term.
This example starts with binary success, without s3's token-cost reward shaping.

## Tests

```sh
LITELLM_LOCAL_MODEL_COST_MAP=True python -m pytest -q examples/async_grpo_harbor_swarm/test_*.py
ruff check examples/async_grpo_harbor_swarm
```

CPU tests use real msgboard processes and TRL packing objects. They check board isolation,
idempotent posts, cursor reads, the bundled container CLI, preserved masks/logprobs, and
equal-team loss/gradient invariance under trajectory forks and chunking. Container/GPU
validation additionally requires the native oracle, two Pi agents, two training updates,
adapter reload, and held-out evaluation; CPU tests do not establish those results.

Validation on 2026-11-19: 24 example/reconciler tests passed. The native Harbor oracle
earned reward 1 on the pinned skeleton task on both CPU and GPU hosts. An isolated
Podman 5.8.8 installation with slirp rootless networking resolved the GPU host's old
system runtime limitation. A two-agent, four-call Pi check then passed native grading
and exact capture validation: 373 supervised tokens, two board messages, and a valid
unsuccessful team reward of 0. The longer run completed its first two optimizer steps
on eight valid teams. Its first group had rewards `[1, 0, 1, 0]`, gradient norm 0.336,
and changed 248 LoRA-B tensors in the saved checkpoint; vLLM loaded policy versions
through `v3`. Final held-out checkpoint evaluation remains outstanding. These are
integration checks, not evidence of benchmark improvement.

`message_board.py` and `board_server.py` are adapted from
[`huggingface/flock`](https://github.com/huggingface/flock/tree/d624786949f498b08733eb622a52233e049464c5/src/flock/harbor/builtin)
with attribution. Both use the same pinned msgboard SDK/server as that implementation.

Full-model extension checks: 21 CPU tests pass, including a two-process Gloo check of
the actual policy loss against a single-process reference. Trainer and NCCL weight-transfer
imports passed on an H100. Optional FSDP2/serving and native concurrency checks run
inside the Slurm allocation when requested; CPU/import checks do not establish their outcome.

The one-node launcher uses NCCL socket transport (`NCCL_NET=Socket`, `NCCL_P2P_DISABLE=1`,
`NCCL_SHM_DISABLE=1`, `NCCL_CUMEM_HOST_ENABLE=0`, `NCCL_IB_DISABLE=1`) for the
host/container weight-transfer boundary and enables NCCL INIT/NET diagnostics.
This conservative setting also applies to intra-trainer collectives and can reduce
throughput. Native training starts directly by default. Monitoring
records explicit queued/running/failed status and the rank-zero failure summary.


For native small-batch validation, set `SWARM_GROUP_SIZE=4`,
`SWARM_TEAMS_PER_UPDATE=4`, `SWARM_CONCURRENT_TEAMS=4`,
`SWARM_MAX_STEPS=2`, and `SWARM_VERIFY_CHECKPOINT=1` before submitting
`full_run.sbatch`. This runs real 16-agent Harbor teams and reloads the saved full
model after training. Equal-reward groups can have zero policy gradient; the run
records `mixed_groups` and `parameters_changed` without presenting an optimizer
call as evidence of learning. `SWARM_DATASET_REVISION` pins a task repair commit.
The launcher defaults `SWARM_DISTRIBUTED_TIMEOUT` to 43,200 seconds for long runs;
`SWARM_RUN_SECONDS` must match the requested Slurm allocation duration.

The full-node launcher uses a 32,768-token context and packing budget. Pi 1.4.0
reserves 4,096 context tokens; with compaction disabled, a 16k window caused native
agents to finish with one-token, length-limited responses after about 27 calls in
validation. `SWARM_MAX_CONTEXT` sets the same limit in Pi and vLLM;
`SWARM_TOKEN_BUDGET` must be at least that large. The 30-minute and 64-call limits
remain upper bounds. Increasing context provides more room to solve tasks; it does
not establish a positive reward or learning improvement.
