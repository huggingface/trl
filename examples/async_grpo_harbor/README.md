# AsyncGRPO with Harbor

An installed agent runs each task through OpenEnv. TRL trains on the engine's captured token IDs,
logprobs, and loss masks, then updates the same vLLM instance.

## Local setup

Use Python 3.12 and this TRL checkout. The typed capture API was merged in
[OpenEnv #1280](https://github.com/huggingface/OpenEnv/pull/1280). Install the pinned upstream revision:

```sh
git clone https://github.com/huggingface/OpenEnv.git OpenEnv
git -C OpenEnv checkout --detach 86a180ede21e044f7929b9a7783ad83aa67d83a3
pip install -e . trackio datasets ./OpenEnv
export PYTHONPATH="$PWD/OpenEnv/envs${PYTHONPATH:+:$PYTHONPATH}"
```

The source path is required because this revision's environment wheel does not package
`harbor_env.harness`. Both the core package and the harness above use the same revision.

Run the OpenEnv server with its sandbox credentials, then start vLLM and the trainer using the commands
in [async_grpo_harbor.py](async_grpo_harbor.py). The server additionally needs `pip install './OpenEnv[harbor]'`;
the trainer can use an existing server. Use the same model and vLLM endpoint for rollouts and weight updates.
Set the server's `MAX_CONCURRENT_ENVS` to at least `--max-inflight + 1` (9 for the defaults); the factory
keeps one connection for task metadata. The Jobs launcher sets this capacity automatically.

Training saves a checkpoint every 50 steps, keeps the latest three, and saves the final model and
tokenizer in the output directory on successful completion, including runs shorter than 50 steps.

## Hugging Face Jobs

[launcher.py](launcher.py) starts the server, vLLM, and trainer on one two-GPU job. It installs the pinned
OpenEnv runtime and checks out the matching harness source. Pass `HF_TOKEN` and the sandbox credential
as job secrets. A bucket mounted at `/data` preserves checkpoints and logs.

Use `--train-script-url` with the raw URL of the TRL revision being tested before the example is merged.
Use `--tunnel-check-s 0` to disable tunnel supervision. A server restart interrupts existing sessions;
subsequent sessions receive the new proxy URL.

## Capture contract

- OpenEnv supplies a validated `TrainingTrace` with required masks. Prompt and completion IDs come
  from the inference engine; TRL does not re-tokenize captured prompts or select agent turns.
- `loss_mask` covers prompt plus completion. Prompt positions are zero; completion positions may mix
  zeros and ones. TRL maps the completion span to its generic `TurnRecord.output_mask`.
- For example, prompt `[10, 11]`, completion `[12, 13, 14]`, and mask `[0, 0, 1, 0, 1]` train on tokens
  `12` and `14`, while retaining `13` as context.
- The worker constructs the session factory with its sampling policy. Pass a callable such as
  `partial(HarborSessionFactory, server_url, ...)`; do not configure sampling separately on the factory.
  OpenEnv applies the policy to captured requests.
- Invalid captures stop the worker. Transport failures remain unscorable; an agent timeout retains
  valid captured turns and the verifier's score.

Execution status and structured tool outcomes are follow-up OpenEnv API work. Today, TRL recognizes
timeouts through `TimeoutError` and reconstructs conversation and tool diagnostics from captures.
Set agent turn and time budgets on the session factory; the worker only forwards sampling.

Rewritten histories can produce multiple training rows. Token retention does not guarantee equal
rollout weighting or a suitable memory budget for every harness.

Run the packaged contract and reconciliation tests with this checkout on `PYTHONPATH`:

```sh
python -m pytest tests/experimental/test_openenv_tito.py \
    tests/experimental/test_async_grpo_trainer.py::TestReconciler -q
```

## Single harness + Single GPU with LoRA on Qwen3.5-0.8B

The example below shows how to train a small model, i.e., Qwen3.5-0.8B, with single harness RL on a single GPU, leveraging LoRA and IPC communication to synchronise the trained weights with the vLLM server. In this example, we were able to make the model converge after ~200 optimization steps. The sandboxes are executed with HF jobs instead of E2B. The mi

### Setting up

Make sure to subscribe to HF pro to be able to use HF jobs. Make also sure to create a HF bucket in order to save the intermediate artifacts and final checkpoints.

### Commands

The minimum required GPU VRAM is 24GB. We tested this setup on 2 different hardwares

- 1x NVIDIA H200 GPU: Training time ~2.5h
- 1x NVIDIA L4 GPU: Training time ~12h 

Taking into account the CPU sandboxes, the run should cost you below 20$.

The commands only differ by the GPU flavor and the attention implementation: the H200 uses Flash Attention 3, while the L4 doesn't support it and uses Flash Attention 2.

<details>
<summary>1x NVIDIA H200</summary>

```sh
hf jobs uv run --flavor h200 --image huggingface/trl --secrets HF_TOKEN --timeout 12h \
    --volume hf://buckets/<user>/<bucket>:/outputs \
    https://raw.githubusercontent.com/huggingface/trl/main/examples/async_grpo_harbor/launcher.py -- \
    --data-root /outputs \
    --train-script-url https://raw.githubusercontent.com/huggingface/trl/main/examples/async_grpo_harbor/async_grpo_harbor.py \
    --model Qwen/Qwen3.5-0.8B --harness pi --split FineEnvs/SmolDataEnvs-harbor-train --n-tasks 1000 --difficulty easy \
    --sandbox hf-sandbox --docker-image docker.io/savatar101/env-data-agent-train:base \
    --vllm-device 0 --train-device 0 --weight-transfer ipc --gpu-memory-utilization 0.3 --token-budget 65536 \
    --max-inflight 32 --max-steps 200 --save-steps 25 --save-total-limit 0 --lora-r 32 --learning-rate 2e-5 \
    --attn-implementation kernels-community/flash-attn3 \
    --num-generations 8 --temperature 0.8 --max-completion-length 16384 --agent-step-limit 17 --agent-timeout 600 \
    --per-device-train-batch-size 4 --gradient-accumulation-steps 4
```

</details>

<details>
<summary>1x NVIDIA L4</summary>

```sh
hf jobs uv run --flavor l4x1 --image huggingface/trl --secrets HF_TOKEN --timeout 12h \
    --volume hf://buckets/<user>/<bucket>:/outputs \
    https://raw.githubusercontent.com/huggingface/trl/main/examples/async_grpo_harbor/launcher.py -- \
    --data-root /outputs \
    --train-script-url https://raw.githubusercontent.com/huggingface/trl/main/examples/async_grpo_harbor/async_grpo_harbor.py \
    --model Qwen/Qwen3.5-0.8B --harness pi --split FineEnvs/SmolDataEnvs-harbor-train --n-tasks 1000 --difficulty easy \
    --sandbox hf-sandbox --docker-image docker.io/savatar101/env-data-agent-train:base \
    --vllm-device 0 --train-device 0 --weight-transfer ipc --gpu-memory-utilization 0.3 --token-budget 65536 \
    --max-inflight 32 --max-steps 200 --save-steps 25 --save-total-limit 0 --lora-r 32 --learning-rate 2e-5 \
    --attn-implementation kernels-community/flash-attn2 \
    --num-generations 8 --temperature 0.8 --max-completion-length 16384 --agent-step-limit 17 --agent-timeout 600 \
    --per-device-train-batch-size 4 --gradient-accumulation-steps 4
```

</details>