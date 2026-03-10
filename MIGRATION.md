# Migrating from TRL v0 to v1

This guide covers the breaking changes introduced in TRL v1 and how to update your code. Most structural changes (trainers moved to experimental, removed model classes, etc.) already shipped in v0.29 — if you're already on v0.29, this migration is minimal.

## Changed defaults

| Config | Parameter | v0 default | v1 default | Action needed |
| --- | --- | --- | --- | --- |
| `GRPOConfig` | `vllm_mode` | `"server"` | `"colocate"` | If you use `use_vllm=True` without specifying `vllm_mode`, vLLM will now run in the same process instead of connecting to a separate server. Set `vllm_mode="server"` explicitly if you rely on server mode. |
| `RLOOConfig` | `vllm_mode` | `"server"` | `"colocate"` | Same as above. |
