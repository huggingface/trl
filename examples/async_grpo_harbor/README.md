# AsyncGRPO with Harbor

An installed agent runs each task through OpenEnv. TRL trains on the engine's captured token IDs,
logprobs, and loss masks, then updates the same vLLM instance.

## Local setup

Use Python 3.12 and this TRL checkout. Pin OpenEnv to the merge of
[OpenEnv #1036](https://github.com/huggingface/OpenEnv/pull/1036):

```sh
git clone https://github.com/huggingface/OpenEnv.git OpenEnv
git -C OpenEnv checkout --detach 34825a772ae54760fb6bf7a8b2073a4a85714004
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

Prebuild cold E2B task templates with serial rollouts before concurrent training; simultaneous first
builds can fail. The Jobs launcher warms task 0 only, so other task-specific templates may need warming.

## Hugging Face Jobs

[launcher.py](launcher.py) starts the server, vLLM, and trainer on one two-GPU job. It installs the pinned
OpenEnv runtime and checks out the matching harness source. Pass `HF_TOKEN` and the sandbox credential
as job secrets. A bucket mounted at `/data` preserves checkpoints and logs.

Use `--train-script-url` with the raw URL of the TRL revision being tested before the example is merged.
Use `--tunnel-check-s 0` to disable tunnel supervision. A server restart interrupts existing sessions;
subsequent sessions receive the new proxy URL.

## Capture contract

- Prompt and completion IDs come from the inference engine; TRL does not re-tokenize captured prompts.
- `loss_mask` covers prompt plus completion. Prompt positions are zero; completion positions may mix
  zeros and ones. TRL maps the completion span to its generic `TurnRecord.output_mask`.
- For example, prompt `[10, 11]`, completion `[12, 13, 14]`, and mask `[0, 0, 1, 0, 1]` train on tokens
  `12` and `14`, while retaining `13` as context. A whole-turn filter cannot express this selection.
- Capture and trainer sampling must agree. The producer fills full-vocabulary defaults; missing or
  mismatched effective-policy metadata is rejected.
- Invalid captures stop the worker. Transport failures remain unscorable; an agent timeout retains
  valid captured turns and the verifier's score.

Rewritten histories can produce multiple training rows. Token retention does not guarantee equal
rollout weighting or a suitable memory budget for every harness.

Run the CPU contract and launcher checks with the pinned checkout on `PYTHONPATH`:

```sh
python -m pytest tests/experimental/test_openenv_tito.py tests/experimental/test_harbor_example.py \
    tests/experimental/test_async_grpo_trainer.py::TestReconciler tests/test_vllm_control_requests.py -q
```
