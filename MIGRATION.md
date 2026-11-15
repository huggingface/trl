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

## Removed automatic `None` stripping from trainer preprocessing

TRL trainers (SFT, DPO, Reward) no longer automatically strip `None` values from dataset examples during preprocessing. Previously, each trainer applied `remove_none_values` via `dataset.with_transform` to work around tabular backends (Arrow/Parquet) inserting `None` for missing keys in nested structures.

This affects datasets that contain `None` values because they were:

- Created before `datasets` v4.7.0, which introduced the Json dtype that preserves nested structures without inserting `None`.
- Created with `datasets` v4.7.0 or later, but saved without using the Json feature.

**Action needed:** If your dataset falls into one of the above categories and contains `None` values in nested columns, apply the fix manually before training:

```python
from trl.trainer.utils import remove_none_values

dataset = dataset.with_transform(remove_none_values)
trainer = SFTTrainer(..., train_dataset=dataset)
```

Datasets created or re-saved with `datasets` v4.7.0+ using the Json dtype are unaffected.

## Migrating from an earlier version

Depending on which version you're migrating from, refer to the [release notes](https://github.com/huggingface/trl/releases) for v0.29 and earlier for version-specific changes.
