# Usage Stats Collection

TRL collects anonymous usage statistics to help the maintainers understand which trainers, model architectures, and hardware configurations are used in the wild. This data informs prioritization decisions about which features to invest in and which to deprecate.

## What is collected

A single ping is sent each time a trainer is instantiated. The payload contains:

- TRL version (e.g. `1.5.0`)
- Trainer class name (e.g. `SFTTrainer`, `GRPOTrainer`)
- Model architecture (`model.config.model_type`, e.g. `llama`, `qwen3`)
- Whether [PEFT](https://github.com/huggingface/peft) is in use
- Distributed backend (`deepspeed`, `fsdp`, `ddp`, or `none`)
- World size, bucketed (`1`, `2-8`, `9-64`, `65+`)
- Accelerator type (`cuda`, `xpu`, `npu`, `mlu`, `mps`, `cpu`)
- GPU model name (e.g. `NVIDIA H100 80GB HBM3`), when available

No dataset names, file paths, model identifiers, hyperparameter values, or any other user-provided data are collected. As with any HTTP request, the source IP and standard HTTP headers are visible to the server.

Telemetry is not sent in CI environments (i.e. when the `CI` environment variable is set), nor in offline mode.

## How to opt out

Set either of the following environment variables to disable telemetry:

```bash
export HF_HUB_DISABLE_TELEMETRY=1   # disables telemetry for all HF libraries
export HF_HUB_OFFLINE=1             # disables all network calls to the Hub
```

## Why this helps

Without usage data, the maintainers have no way to know which parts of the library matter to users. Telemetry lets us answer questions like:

- Which trainers are widely adopted, and which can be deprecated safely?
- Which model architectures are most often fine-tuned with TRL, so we can prioritize their support?
- What hardware configurations are users running on, so we can test and optimize for them?

If you find TRL useful, leaving telemetry enabled is a low-cost way to help us make it better.
