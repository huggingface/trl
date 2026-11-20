# W2 on sandbox backends each library ships

Question: TRL runs W2 on HF Sandboxes through OpenEnv's Harbor server, and no other library has an HF Sandboxes backend. Under the benchmark rule (user, 2026-11-10: each library runs only with what it ships, no custom losses or patches; a backend the library already supports is fine, recorded per library), which backend can prime-rl, miles, verl and SkyRL use for the same TermGrade tasks and terminus-2 budgets?

Short answer: **local containers on our own nodes work**, without root, through two backends the libraries already ship:

- **Docker API on rootless podman** (verifiers' `docker` runtime, Harbor's `docker` environment): the docker CLI talks to `podman system service` on a socket under `/scratch`. prime-rl's native path (verifiers `HarborTaskset` + `terminus_2` harness + docker runtime) ran 8 terminus-2 episodes end to end against Qwen3.5-9B on a GPU node (job 365405, mean reward 0.25), with two wrappers outside the library: `unshare -r` around the verifiers client and a `docker run --rm` shim for the GPU nodes' old podman 3.4. Harbor's `docker` env works on the CPU nodes' podman 4.9 but not on the GPU nodes' podman 3.4. miles refuses `docker` in its in-process Harbor path.
- **Harbor's `singularity` environment on an unprivileged Apptainer** (official release .deb extracted, no setuid): terminus-2 with the W2 kwargs ran 8 trials end to end on a GPU node (job 365419, 0 exceptions, mean reward 0.25), with the shipped kwarg `singularity_no_mount=home,tmp`. SkyRL's Harbor pin and miles' `harbor-miles-v0.20.0` branch both carry this env; it is miles' only local option. Not yet run inside miles' container.

verl ships no Harbor / terminal-task integration and no sandbox backend for one, so W2 isn't runnable on verl as shipped on any backend.

The sandbox backend then differs per library (HF Sandboxes for TRL, local podman or Apptainer for the others), so sandbox start and exec latency become a per-library variable to report.

## 1. Backends each library supports as shipped

Paths are under `/fsx/amine_dirhoussi/stable-bench-src/`. `vf/` = `prime-rl/deps/verifiers/verifiers/v1/`. The Harbor revs miles and SkyRL pin are read from the Harbor clone at `stable-bench-src/harbor` (`git show <rev>:...`).

| library | how it runs a Harbor/terminal task | backends shipped | local (no daemon, no root) option |
|---|---|---|---|
| **prime-rl** | verifiers `HarborTaskset` (`vf/tasksets/harbor/taskset.py`) + a harness (`terminus_2`, `bash`, ...) that runs **inside** the box and reaches the policy through verifiers' interception server | `prime` (Prime Sandboxes, `vf/runtimes/backends/prime.py`, needs `PRIME_API_KEY`), `docker` (`vf/runtimes/backends/docker.py`: docker CLI, `docker run --detach --network bridge`, then `docker exec`; `DOCKER_HOST` honoured; `is_local=True` so no tunnel), `subprocess` (no task images, so not usable for TermGrade), `custom` (plugin point for your own `RuntimeBackend`, i.e. new code) | `docker` against rootless podman (tested, works with one wrapper, see 2) |
| **miles** | `examples/experimental/harbor`: Harbor `Trial.run()` in-process in the rollout worker, terminus-2 host-side; Harbor from `harbor-framework/harbor@harbor-miles-v0.20.0`; backend = `HARBOR_ENV_TYPE`, extra kwargs = `HARBOR_ENV_KWARGS` | any Harbor `EnvironmentType` at that branch (docker, daytona, e2b, modal, runloop, langsmith, ec2, gke, ack, openshift, novita, apple-container, **singularity**, islo, tensorlake, cwsandbox, wandb, use-computer, cua-cloud, blaxel, opensandbox, beam, skypilot). Credentials wired only for daytona, e2b, modal (`miles/rollout/agentic/credentials.py`). **`docker` is rejected** in the in-process path (`launch_common.py:69-72`: "docker needs a Docker daemon next to Trial.run(); use examples/swe-agent-harbor-docker"). Verified providers in the docs: AgentENV, Daytona, E2B, Modal, Novita. Also `examples/swe-agent-harbor-docker` (separate agent server on a Docker host, GLM-4.7 tool-call setup, not terminus-2) and `examples/experimental/openenv` (OpenEnv tbench2 env on Daytona / E2B / Modal / Novita, or one shared env server via the Docker SDK, `network_mode="host"`; tbench2, not Harbor/terminus-2) | Harbor `singularity` (tested at Harbor 0.25.0, same kwargs at miles' rev; not yet run inside the miles container) |
| **verl** | none. `verl/experimental/agent_loop` is a generic loop API (would need our own `AgentLoop` = new code); `SandboxFusion` is a code-execution reward scorer; verl-recipe `swe_agent` runs SWE-agent through SWE-ReX `deployment.type="docker"` with `/var/run/docker.sock` mounted (SWE-bench style, not Harbor, not terminus-2) | no Harbor / terminal-task backend | none for W2 as specified |
| **SkyRL** (for reference; its W2 is already reported as not runnable stock for loss/group-zeroing reasons) | `examples/train_integrations/harbor`: Harbor in-process, terminus-2; Harbor pinned to git `3de07a0e` (v0.13.1+25), extras `[daytona,modal]` | docker, daytona (default in `default.yaml`), e2b, modal, runloop, langsmith, gke, novita, apple-container, **singularity**, islo, tensorlake, cwsandbox, wandb, use-computer. No hf-sandbox (added in v0.21.0), no podman env (added later) | Harbor `docker` on rootless podman, or Harbor `singularity` |

Harbor 0.25.0 (TRL's, installed in `bench-trl-w2` / `bench-skyrl-w2`) additionally has `podman` (uses `podman compose`, not installed here), `hf-sandbox`, `prime`, `kata`, `sbx` and more; none of the four libraries pins it.

## 2. Local containers on our nodes

### What the nodes have (job 365369 on hopper-cpu, 365394 on hopper-dev)

| | hopper-cpu (Ubuntu 24.04) | GPU nodes, hopper-dev / hopper-b200 (Ubuntu 22.04) |
|---|---|---|
| Docker daemon | no (no `/var/run/docker.sock`) | no daemon for users (socket permission denied) |
| docker CLI + compose plugin | not installed | `docker` 29.x + compose v5 installed |
| podman / buildah | podman **4.9.3**, buildah 1.33 (rootless, overlay on `/scratch/containers/<uid>`, crun, netavark, cgroup v2) | podman **3.4.4** (rootless, slirp4netns) |
| subuid range | `amine_dirhoussi:893216:65536` | same |
| unprivileged user namespaces | on (`apparmor_restrict_unprivileged_userns = 0`) | on |
| Apptainer / Singularity | not installed | not installed |
| Enroot + Pyxis | yes | yes (no library has an enroot backend) |

What we added, all user-space, all under `/fsx/amine_dirhoussi/stable-bench-src/container-tools/`:

- `bin/docker` (official static docker CLI 29.9.3 from download.docker.com) and `docker-config/cli-plugins/docker-compose` (compose v5.7.0 from docker/compose releases): clients only, for nodes without them.
- `podman-env.sh`: sets `XDG_RUNTIME_DIR` under `/scratch`, `DOCKER_HOST=unix://.../podman.sock`, and `podman_service_start` (`podman system service --time=0`). Source it in the job, call `podman_service_start`, and every docker client (Harbor's `docker compose`, verifiers' `docker run/exec`) talks to rootless podman.
- `registries.conf` (`CONTAINERS_REGISTRIES_CONF`): Docker Hub's anonymous pull limit is exhausted from the cluster NAT (`toomanyrequests` in job 365369), so `docker.io` official images are pulled from the ECR Public mirror `public.ecr.aws/docker`.
- `images/termgrade-base.tar` (1.22 GB, `podman save`): `workloads/agentic/base_image/Dockerfile` built verbatim with podman (30 s, job 365382), tagged `localhost/termgrade-base:latest`. Each job `podman load`s it (~10 s). `images/termgrade-base.sif` (407 MB): the same image converted for Apptainer (`apptainer build ... docker-archive:termgrade-base.tar`).
- `apptainer/`: Apptainer 1.5.5 extracted from the official `apptainer_1.5.5_amd64.deb` (`dpkg-deb -x`, no setuid), with `usr/etc -> ../etc` and `usr/var -> ../var` so the relocated binary finds its config, and per-OS missing libs fetched with `apt-get download` (24.04 lacked only `libtalloc2`), see `stable-bench-runs/native-sandbox/scripts/apptainer_fix.sh`. `bin/singularity -> apptainer` because Harbor calls `singularity`.

Task dirs: `stable-bench-src/termgrade/harbor-local/{train,eval}` (for docker/podman) and `harbor-local-sif/{train,eval}` (for singularity) are copies of the prepared `harbor/{train,eval}` dirs with only `docker_image` changed (to `localhost/termgrade-base:latest` and to the `.sif` path). The HF Space name `hf.co/spaces/aminediroHF/termgrade-base` can't be used locally: uppercase is not valid in a docker image reference, and the image is the same Dockerfile anyway. `environment/`, the healthcheck (`prepare.sh` = setup.sh), tests, timeouts and instructions are unchanged.

### Results

All on hopper-cpu unless noted. Scripts and logs in `/fsx/amine_dirhoussi/stable-bench-runs/native-sandbox/{scripts,logs,runs}`.

| test | job | result |
|---|---|---|
| runtime probe | 365369 | rootless podman 4.9.3 works; Docker Hub pulls rate-limited |
| build base image with podman, docker CLI on podman socket, `docker run --network bridge --cpus 1 --memory 2g` | 365382 | works; image 1.22 GB, built in 30 s |
| **Harbor `docker` env** (Harbor 0.25.0 `harbor run -e docker -a oracle`) on 2 TermGrade tasks | 365382 | **reward 1.0 on both**, 15 s and 59 s per trial (container start + upload of `environment/` + healthcheck setup.sh + oracle + verifier) |
| **verifiers `docker` runtime** (prime-rl's `vf-validate harbor --runtime.type docker`, model-free noop check: setup + healthcheck + test staging + verifier) | 365382 | **fails as is**: `tar: test.sh: Cannot change ownership to uid 150231 ... Invalid argument` at test staging (see below) |
| same, client wrapped in `unshare -r` | 365384 | **works**: noop check valid on both tasks (reward 0 on the untouched task, as expected), 25-29 s per task. The gold check is `unchecked`: `HarborTask` has no `validate` hook, so the oracle solution isn't exercised by vf-validate |
| **Harbor `singularity` env** (unprivileged Apptainer, `.sif` image, `-a oracle`) | 365385, 365389 (failed), **365393** | **reward 1.0 on both** with `--ek singularity_no_mount=home,tmp`; 17-18 s environment setup per trial (the in-box FastAPI server bootstraps a venv with pip each start) |
| GPU node OS (Ubuntu 22.04, podman 3.4.4) | 365394 (hopper-dev) | `docker run --rm` through podman 3.4's compat API never returns after the container exits (killed by hand). Detached runs + exec work |
| **model in the loop, prime-rl's native path**: vLLM Qwen3.5-9B (thinking off) on 1 GPU + `unshare -r vf-run harbor --env.agent.harness.id terminus_2 --env.agent.runtime.type docker --env.agent.max-turns 30 --env.agent.max-turn-tokens 8192 --env.agent.timeout.rollout 600`, 4 tasks x 2 | 365394 (hung, see 3 below), **365405** (hopper-dev, with the `--rm` shim) | **8/8 episodes ok, mean reward 0.25**, 158 model calls through verifiers' interception server, no tunnel. Per episode: box boot ~4.2 s, task setup (upload `environment/` + healthcheck setup.sh) 3-4 s, agent 1.5-5.6 min, scoring ~1.3 s, teardown ~1.4 s. 4 of 8 stopped at the 30-turn cap (`stop=max_turns`) |
| model in the loop, Harbor `docker` env + terminus-2 | 365405 (hopper-dev) | **fails on podman 3.4**: `CNI network "<project>_default" not found` when compose starts the container (see 3 below). Works on podman 4.9 (oracle, 365382) |
| **model in the loop, Harbor `singularity` env + terminus-2** with the W2 kwargs (`configs/skyrl/w2-M.harbor.yaml`'s: 30 turns, summarisation off, 8192 output tokens, 65,536 context, 600 s), 4 tasks x 2 | **365419** (hopper-dev, Apptainer 1.4.5 for Ubuntu 22.04, job 365418 checks it) | **8/8 trials, 0 exceptions, mean reward 0.25** (pass@2 0.5). One trial: environment setup 12.1 s (Apptainer start + server bootstrap + healthcheck), agent setup 1.6 s, verifier 1.0 s |
| **load on a B200 node** (ip-10-53-7-103, Ubuntu 22.04, podman 3.4.4, 1 GPU + 64 CPUs of the node) | **365422** (waiter 365390, 18.7 min) | (1) 64 idle task containers: host memory unchanged within noise (57.7 GB before, 57.3 GB with 64 up). (2) Harbor singularity, oracle, 28 train tasks x 3 at 64 concurrent: 84 trials in 67 s, reward 1.0 on 80, 0.0 on 3, 1 `RuntimeError` (`Failed to upload directory: cp: cannot stat '/staging/environment'`). (3) **prime-rl W2 shape**: terminus-2 on the verifiers docker runtime, 8 tasks x 8 episodes, 64 in flight against one vLLM GPU: **63/64 episodes ok, mean reward 0.27**; 37 hit the 30-turn cap, 26 finished on their own; agent time mean 252 s (max 413 s), of which model 196 s; box boot mean 15.7 s (p90 21.3 s) and task setup 9.6 s (p90 10.8 s) at 64-way start, scoring 1.3 s. The one failure: `rootlessport listen tcp 127.0.0.1:<port>: bind: address already in use` at `docker run` (port race when verifiers publishes its service port under podman 3.4's rootlessport). Host memory peaked at 80.8 GB (+23 GB incl. vLLM's host side), load average 21.6 on 64 CPUs. (4) Harbor singularity + terminus-2, 4 tasks x 2: 8 trials, 0 exceptions, mean 0.125 |

Four things found on the way, each with a no-patch workaround:

1. **verifiers' test staging breaks on rootless engines** (prime-rl). `HarborTask.stage_tests` packs `tests/` with Python's `tarfile` (owner = host uid 150231 in the headers) and unpacks it in the box with `tar -xzf` as root, which restores ownership. In a rootless container uid 150231 isn't mapped (container root = our uid, container 1-65536 = our subuid range), so `chown` fails with EINVAL. The same file's `setup()` unpacks `environment/` with `--no-same-owner`; `stage_tests` doesn't. It's the same bug class as the hf-sandbox tar patch, but here it needs no patch: running the verifiers client (vf-run / prime-rl's orchestrator) under `unshare -r` maps our uid to 0 in a user namespace, so `tarfile` writes uid 0 and the box's `tar` chowns to root. Network, files and the podman socket are unaffected. Upstream-reportable (one flag).
2. **Harbor's singularity default drops DNS here.** Its default `singularity_no_mount = "home,tmp,bind-paths"` also removes the bind of `/etc/resolv.conf` on this Apptainer, so the in-box bootstrap can't resolve `bootstrap.pypa.io` and dies with `FATAL: cannot bootstrap pip` (the TermGrade image's `/usr/bin/python3` is Debian's 3.11 without ensurepip). `singularity_no_mount=home,tmp` is a shipped env kwarg (Harbor CLI `--ek`, miles `HARBOR_ENV_KWARGS`, SkyRL `environment.kwargs`) and fixes it (job 365392 compares the variants).
3. **podman 3.4.4 on the GPU nodes** (Ubuntu 22.04; the CPU nodes' 24.04 has 4.9.3) has two gaps. (a) An attached `docker run --rm` never returns after the container exits (SIGTERM doesn't stop the client either). verifiers runs a short-lived `--rm` helper per box to plant its egress-proxy listener in the box's network namespace, so every episode hung at start (365394). Fixed with a client-side shim, `container-tools/shim/docker`: for an attached `run --rm` it runs without `--rm` (`--cidfile`) and removes the container itself; everything else goes straight to the real CLI. `podman-env.sh` puts it on PATH only when podman is 3.x. (b) Rootless podman 3.4 uses CNI and can't attach containers to compose-created networks, so Harbor's `docker` env fails on GPU nodes (365405); Harbor-based libraries use the `singularity` env there instead. A newer podman on the GPU nodes would remove both, but the only no-root route found is a third-party static build (mgoltzsche/podman-static), which I didn't install (it needs your OK, or a cluster-side podman upgrade via #science-cluster-support).
4. **Apptainer's 1.5.5 .deb is built for Ubuntu 24.04** (needs libsubid4 and fuse 3.12), so `container-tools/apptainer-env.sh` picks the newest release whose helpers resolve on the node: 1.5.5 on 24.04 (CPU nodes), 1.4.5 on 22.04 (GPU nodes). The same `.sif` works with both.

### Resources per sandbox

Measured at 64 on a B200 node (365422, `runs/b200-365422/host.csv`): 64 idle boxes cost no measurable host memory; 64 live terminus-2 episodes plus vLLM peaked at +23 GB host memory and a load average of 21.6 on the job's 64 CPUs. Sandbox start slows from ~4 s to ~16 s (p90 21 s) and setup from 3-4 s to ~10 s when 64 start at once. Background: the image is 1.22 GB on disk (overlay, shared by every container) or 407 MB as a `.sif`; TermGrade's `setup.sh` scripts are small (the pool excludes apt/curl/wget), and an idle box is one `sleep infinity` (docker) or one uvicorn server (singularity). With 2 TB of RAM on a B200 node, 64 boxes at HF `cpu-basic` size (2 vCPU / 16 GB) would be 128 vCPU / 1 TB if every box used its full allowance, which they don't. The risk is CPU, not memory: setup.sh bursts and test runs share the node with the trainer's dataloader and vLLM's API servers. Caps exist where the backend has them: verifiers' docker runtime takes `cpu`/`memory` (from the task or `DockerConfig`); Harbor's singularity env sets none by design (it relies on the Slurm job's limits).

## 3. Cloud backends: what we'd need

Nothing was signed up for or spent. None of these credentials exist in the environment (`DAYTONA_*`, `MODAL_*`, `E2B_*`, `PRIME_*`, `RUNLOOP_*` unset, no `~/.modal.toml` or Prime config).

Rough cost for one W2 run: 64 live sandboxes for ~60 steps. At TRL's ~400-700 s per step that's ~10 h, so ~640 sandbox-hours; with 1 vCPU and 2 GiB each (HF `cpu-basic` is 2 vCPU / 16 GB, so this is smaller), plus the eval passes. Rates from the providers' pricing pages and third-party comparisons on 2026-11-10; check before relying on them.

| backend | who ships it | what the user must provide | list rate | ~cost per W2 run (640 sandbox-h at 1 vCPU / 2 GiB) |
|---|---|---|---|---|
| Prime Sandboxes | prime-rl (verifiers default runtime) | Prime Intellect account + `PRIME_API_KEY` (team id optional); the box reaches the model through Prime's tunnel, so the run also depends on that tunnel | $0.02 / vCPU-h + $0.0125 / GiB-h + $0.0002 / GiB-h disk (launch rate through 2026-12-22; another docs page says $0.05 / core-h) | ~$30 (launch rate) |
| Daytona | miles (credentials wired), SkyRL (default), Harbor | Daytona account + `DAYTONA_API_KEY`; default tier caps 4 vCPU / 8 GB / 10 GB disk per sandbox; org concurrency limits apply | $0.0504 / vCPU-h + $0.0162 / GiB-h | ~$55 |
| Modal | miles (wired), SkyRL extra, Harbor | Modal account + token id/secret (`modal token new`) | ~$0.047-0.14 / core-h for sandboxes (non-preemptible sandbox premium; page changed Oct 2026) + memory | ~$40-110 |
| E2B | miles (wired), Harbor | E2B account + `E2B_API_KEY`; the Hobby tier caps at 20 concurrent sandboxes, so 64 needs Pro ($150/month) | ~$0.05 / vCPU-h + memory | ~$50 + $150/month plan |
| Runloop | Harbor (SkyRL / miles revs) | Runloop account + API key | not checked | not estimated |

Costs exclude egress and failed/orphaned sandboxes (the HF Sandbox runs showed killed drivers leak sandboxes; every provider needs the same sweep).

## 4. Recommendation per library

Every path below runs the same prepared TermGrade tasks (only `docker_image` points at the local copy of the same base image), the same healthcheck setup, the same verifier, and terminus-2. What differs per library is where the sandbox lives and, for prime-rl, where terminus-2 runs.

| library | cheapest native path | status | what's left | effort |
|---|---|---|---|---|
| **prime-rl** | verifiers `HarborTaskset` + `terminus_2` harness + **`docker` runtime on rootless podman** on the training node | end to end with a model on GPU nodes (365405; 64 in flight on B200 in 365422); model-free on a CPU node (365384) | (1) launcher: `source container-tools/podman-env.sh; podman_service_start; podman load -i images/termgrade-base.tar`; (2) dataset: `~/.cache/harbor/termgrade-local -> harbor-local/train/tasks` (or per-split names); (3) run the orchestrator (not the trainer or vLLM) under `unshare -r`; (4) W2 budgets via `env.agent.max_turns=30`, `max_turn_tokens=8192`, `timeout.rollout=600`, sampling 1.0/1.0, thinking off on the inference server. The 64-in-flight W2 shape already ran on a B200 node (365422: 63/64 ok, the one failure a rootlessport bind race) | **~4-6 h** to a smoke run, then the 60-step run |
| **miles** | Harbor in-process, **`HARBOR_ENV_TYPE=singularity`**, unprivileged Apptainer | terminus-2 + singularity env works end to end on Harbor 0.25.0 with the W2 kwargs (365419); not yet inside miles | (1) install `harbor @ git+...@harbor-miles-v0.20.0` where miles' rollout worker imports it (like math-verify: `--target` dir on PYTHONPATH, since miles runs from its image); (2) make `container-tools/apptainer-env.sh` and the `.sif` visible inside miles' Pyxis container (mount `/fsx` and `/scratch`) and check Apptainer runs nested in the Enroot container (user namespaces inside a user namespace: untested, the main risk); (3) `HARBOR_ENV_KWARGS='{"singularity_no_mount":"home,tmp"}'`, tasks from `harbor-local-sif`; (4) budgets: `HARBOR_AGENT_MAX_ITERATIONS=30`, `HARBOR_TERMINUS_2_ENABLE_SUMMARIZE` off, `AGENT_TIMEOUT=600`, `AGENT_MAX_OUTPUT_TOKENS=8192`, `HARBOR_MAX_SEQ_LEN=65536` | **~6-10 h** if nested Apptainer works; if it doesn't, miles has no local backend and needs a cloud account (Daytona/E2B/Modal, section 3) |
| **verl** | none shipped; thin AgentLoop on verl's agent-loop API drives the OpenEnv Harbor server (configs/verl/w2_agent_loop.py) | runnable with ~300 lines of glue (2026-11-11, run 371788) | see envs/verl/NOTES.md W2 | 0 |
| SkyRL | Harbor `docker` on podman 4.x (CPU nodes only) or `singularity` | not needed | moot while its stock W2 stays reported as not runnable (group zeroing on agent timeouts, collapsing built-in loss); if it comes back, `environment.type: singularity` + the same kwargs is ~1 h on top of the existing W2 harness | - |

Gaps to record per library (they change what "same harness" means):

- **prime-rl**: terminus-2 runs inside the box (verifiers installs `harbor==0.25.0` there with uv from PyPI at every box start), and its terminus-2 is built with summarisation on (proactive threshold 8000 free tokens) and no knob to turn it off (ledger prime-rl-013). The 30-call cap and 8192 tokens per call are enforced by verifiers' interception layer instead of terminus-2 itself (it worked: `stop=max_turns` at 30). Sandbox: local podman, box start ~4 s, setup 3-4 s.
- **miles**: terminus-2 host-side as in TRL/SkyRL, all W2 knobs available as env vars. Sandbox: local Apptainer, ~12 s environment setup per trial (the in-box FastAPI server's venv bootstrap is part of it).
- **TRL**: HF Sandboxes, ~5.6 s sandbox start + setup in the SkyRL smoke on the same image.

## What the user would need to provide

Nothing for the local paths. Optional, each removes a workaround:

- A podman >= 4 on the GPU nodes (cluster-side, via #science-cluster-support), or an OK to use the third-party static build: removes the `--rm` shim and makes Harbor's `docker` env usable on GPU nodes.
- A larger subuid range (>= 150232 ids) would also remove the `unshare -r` wrapper for prime-rl, since uid 150231 could then be mapped inside containers (ops only).
- For a cloud backend instead: one account and key per provider from section 3 (no spend was made).

## Files

- `stable-bench-src/container-tools/`: `podman-env.sh`, `apptainer-env.sh`, `registries.conf`, `shim/docker`, `bin/docker`, `docker-config/cli-plugins/docker-compose`, `apptainer*/`, `images/termgrade-base.{tar,sif}`.
- `stable-bench-src/termgrade/harbor-local{,-sif}/{train,eval}/tasks/`: task copies for the local backends.
- `stable-bench-runs/native-sandbox/scripts/`: `probe_runtimes.sh`, `e2e_cpu.sh`, `e2e_cpu2.sh`, `e2e_sif.sh`, `sif_debug*.sh`, `sif_jammy.sh`, `apptainer_fix.sh`, `b200_local.sbatch` (stages `probe load sifload model`; `SKIP_A`, `VF_TASKS`/`VF_R`/`VF_C`, `B_TASKS`/`B_K`). Logs in `logs/`, run dirs in `runs/`.
- Ledger rows: prime-rl-012/013/014, miles-015, verl-018, all-018/019/020.
