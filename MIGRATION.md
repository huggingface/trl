# Migrating from TRL v0 to v1

This guide covers the breaking changes introduced in TRL v1 and how to update your code. Most structural changes (trainers moved to experimental, removed model classes, etc.) already shipped in v0.29 — if you're already on v0.29, this migration is minimal.

## Changed defaults

| Config | Parameter | v0 default | v1 default | Action needed |
| --- | --- | --- | --- | --- |
| `GRPOConfig` | `vllm_mode` | `"server"` | `"colocate"` | If you use `use_vllm=True` without specifying `vllm_mode`, vLLM will now run in the same process instead of connecting to a separate server. Set `vllm_mode="server"` explicitly if you rely on server mode. |
| `RLOOConfig` | `vllm_mode` | `"server"` | `"colocate"` | Same as above. |

## Renamed options

| Config | Parameter | v0 value | v1 value | Action needed |
| --- | --- | --- | --- | --- |
| `SFTConfig` | `packing` | `"bfd-requeue"` | `"bfd_split"` | Replace `packing="bfd-requeue"` with `packing="bfd_split"`. The old value will still be accepted for a few versions but will be removed in a future release. |

## Migrating from an earlier version

Depending on which version you're migrating from, refer to the [release notes](https://github.com/huggingface/trl/releases) for v0.29 and earlier for version-specific changes.
