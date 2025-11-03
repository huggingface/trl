# Callbacks

## SyncRefModelCallback

[[autodoc]] SyncRefModelCallback

## RichProgressCallback

[[autodoc]] RichProgressCallback

## WinRateCallback

[[autodoc]] WinRateCallback

## LogCompletionsCallback

[[autodoc]] LogCompletionsCallback

## BEMACallback

[[autodoc]] BEMACallback

## WeaveCallback

[[autodoc]] WeaveCallback

## Experimental Callbacks

Some callbacks have been moved to the experimental submodule due to low usage and potential removal in future versions.

### MergeModelCallback

`MergeModelCallback` has been moved to `trl.experimental.mergekit`. To use it:

```python
from trl.experimental.mergekit import MergeModelCallback, MergeConfig

config = MergeConfig()
merge_callback = MergeModelCallback(config)
trainer = DPOTrainer(..., callbacks=[merge_callback])
```

For more details, see the [mergekit integration documentation](https://github.com/arcee-ai/mergekit).
