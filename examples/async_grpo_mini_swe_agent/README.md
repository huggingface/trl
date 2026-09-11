# LoRA AsyncGRPO of `mini-swe-agent` on SWE-Gym

Trains a real coding agent, [`mini-swe-agent`](https://github.com/SWE-agent/mini-swe-agent), on real GitHub issues from [SWE-Gym](https://huggingface.co/datasets/SWE-Gym/SWE-Gym), with a LoRA policy and [`AsyncGRPOTrainer`](https://huggingface.co/docs/trl/async_grpo_trainer). A port of NeMo Gym's [`mini_swe_agent`](https://github.com/NVIDIA-NeMo/Gym/tree/main/responses_api_agents/mini_swe_agent) responses API agent onto TRL's [loop-owning OpenEnv path](https://huggingface.co/docs/trl/openenv#training-on-harnesses-training-a-real-coding-agent-opencode).

## What runs where

```
 trainer (AsyncGRPOTrainer, FSDP2, LoRA)                          vLLM (--enable-lora)
 ┌────────────────────────────────────────────────┐              ┌──────────────────────────┐
 │ HarnessRolloutWorker                           │              │ Qwen/Qwen3-32B           │
 │  └ MiniSWEAgentSessionFactory.create()         │  chat        │  + trl-policy-vN adapter │
 │      ├ Sandbox.create(image=<instance image>)  │  completions │                          │
 │      ├ mini-swe-agent DefaultAgent ────────────┼─────────────▶│                          │
 │      │    bash tool ──▶ sandbox.run(...)  ─────┼───┐          └──────────────────────────┘
 │      └ verify(): run SWE-bench eval in sandbox ┼───┤   ▲ load_lora_adapter(<output_dir>/.vllm_lora/trl-policy-vN)
 └────────────────────────────────────────────────┘   │   │
                                                      ▼   │
                             HF sandboxes, one per rollout, cpu-basic, started from xingyaoww/sweb.eval.x86_64.<instance>
```

- **The agent owns the loop.** `mini-swe-agent` renders SWE-bench's prompts, calls the model, runs the command it gets back in the sandbox, formats the observation, and repeats until the model submits a patch or hits its step, time or format-error limit. TRL never samples a turn.
- **TRL reads the trace.** The agent's model class talks to vLLM through the OpenAI client and records every call the way OpenEnv's interception proxy would: the request as sent, the response, the generated token ids and their logprobs. `HarnessRolloutWorker` rebuilds one training row per turn from those records, then trains with GRPO.
- **The sandbox is the environment.** Each rollout is one [Hugging Face sandbox](https://huggingface.co/docs/huggingface_hub/guides/sandbox) started from the SWE-Gym image of its instance: the repository at the base commit in `/testbed`, its conda environment already built. The agent's commands run there, and the reward is computed there too.
- **The reward is SWE-bench's.** After the agent stops, the session writes SWE-bench's evaluation script into the sandbox (check the test files out at the base commit, apply the held-out test patch, run the FAIL_TO_PASS and PASS_TO_PASS tests), runs it, and grades the log with the SWE-Gym harness. 1.0 if the instance is resolved, 0.0 otherwise, as in NeMo Gym.
- **The policy is an adapter.** Every weight sync writes the LoRA adapter to `<output_dir>/.vllm_lora/trl-policy-vN` and asks vLLM to load it. The agent's requests name that adapter, because in vLLM's API an adapter *is* a model name and a request naming the base model would be served by the base model.

## Differences from NeMo Gym's agent

Same prompts and templates (`mini-swe-agent`'s `swebench.yaml`), same step limit (250), same binary resolved reward, same dataset. Three things differ:

- The sandbox is a Hugging Face sandbox rather than a local Docker or Singularity container, so no container runtime is needed on the training node and rollouts scale out beyond it.
- The evaluation runs inside the same sandbox the agent worked in, on the final `/testbed`, rather than by re-applying the submitted patch in a fresh container.
- The NeMo fork's `collapse_limit` (a warning when the agent repeats a command) is not in upstream `mini-swe-agent` and is not ported.

## Running

### One node, two GPUs

```sh
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 vllm serve Qwen/Qwen3-32B \
    --host 0.0.0.0 --port 8000 --max-model-len 40960 \
    --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
    --logprobs-mode processed_logprobs --generation-config vllm \
    --weight-transfer-config '{"backend":"nccl"}' \
    --enable-lora --max-lora-rank 32 --max-loras 6

CUDA_VISIBLE_DEVICES=1 HF_TOKEN=... accelerate launch --num_processes 1 \
    examples/async_grpo_mini_swe_agent/async_grpo_mini_swe_agent.py --model Qwen/Qwen3-32B
```

`--enable-auto-tool-choice --tool-call-parser hermes` is required: the agent's only tool is `bash`, called through OpenAI tool calling. `--reasoning-parser qwen3` is for Qwen3 hybrid-thinking models. Thinking is off by default (`--enable-thinking` turns it on); the prompt asks the agent to reason in plain text before each command anyway, and a thinking block per command multiplies the context a 100-step trajectory needs.

The trainer needs `HF_TOKEN` in its environment to create sandboxes. `--max-inflight` bounds how many sandboxes exist at once.

### Hugging Face Jobs, disaggregated

The trainer and the vLLM server run in two Jobs on two machines. The adapter travels through a [Storage Bucket](https://huggingface.co/docs/hub/storage-buckets) both Jobs mount at `/lora`: the trainer writes `trl-policy-vN` into it, the server reads it off its own mount. `lora_proxy.py` runs next to the trainer to add the bearer token the Job's exposed port requires, and retries the adapter load for the few seconds the server's mount takes to see a new directory. The setup is the one measured in [huggingface/trl#7017](https://github.com/huggingface/trl/pull/7017)'s bucket experiments.

```sh
./run_all.sh            # bucket, vLLM Job (h200, one GPU), wait for /health, trainer Job (h200x4)
./run_all.sh --wait     # ...and cancel the vLLM Job when the trainer finishes
./stop.sh               # cancel both now
```

Everything is an environment variable with a default:

```sh
MODEL=Qwen/Qwen3-32B VLLM_REPLICAS=2 TRAIN_FLAVOR=h200x4 ./run_all.sh   # replicas are single-GPU: a failed adapter load hangs a tensor-parallel vLLM
TRAIN_ARGS="--max-steps 200 --num-generations 8 --max-inflight 64" ./run_all.sh
RUN_TAG=r32 ./run_all.sh          # names the bucket directory and the trackio run; rerun to resume
TRL_REF=main ./run_all.sh         # the trl revision the trainer Job installs
```

Checkpoints land in the bucket, so a Job that is restarted with the same `RUN_TAG` resumes from the last one. The final adapter is written to `<output_dir>/final-adapter`.

## What to watch

- **`ratio` ≈ 1.0**: the importance-sampling ratio between the trainer's logprobs and vLLM's. It is the one number that proves the agent's requests were served by the current adapter, that the training rows were rebuilt from the same prompts vLLM rendered (tools, `chat_template_kwargs`), and that the LoRA sync works.
- **`reward`**: the fraction of resolved instances. Qwen3-Coder-30B-A3B starts at about 0.10 on SWE-Gym with this agent (NeMo Gym's number).
- **`rollout/samples_per_rollout`**: training rows per rollout. With Qwen3 and thinking off it equals the turn count: the empty think block in the generation prompt is absent from the history, so each turn is its own row and re-forwards the whole conversation (`sample/forwarded_tokens_mean` far above `sample/trained_tokens_mean`).
- **`rollout/turns_mean`** and **`tools/call_frequency`**: how many steps a rollout takes. The step limit is 250; the context length is what usually ends a trajectory first.
- **`harness_reward` is `NaN` for a rollout** when it could not be scored: the sandbox failed to start or died. Such rollouts are dropped from the group baseline, so a high rate silently shrinks the effective group.
