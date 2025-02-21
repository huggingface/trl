# patch_model_runner.py
import torch
import torch.distributed as dist
from sglang.srt.model_executor.model_runner import ModelRunner as _OriginalModelRunner

# Save the original method so we can call it later.
_original_init_torch_distributed = _OriginalModelRunner.init_torch_distributed


def patched_init_torch_distributed(self, *args, **kwargs):
    # Save the original init_process_group function.
    original_init_pg = dist.init_process_group

    def patched_init_process_group(*pg_args, **pg_kwargs):
        # If device_ids is present, remove it.
        if "device_ids" in pg_kwargs:
            # Optionally, you could log here to note that device_ids was removed.
            # For example: print("Removing device_ids from init_process_group kwargs")
            pg_kwargs.pop("device_ids")
        return original_init_pg(*pg_args, **pg_kwargs)

    # Replace the function temporarily.
    dist.init_process_group = patched_init_process_group
    try:
        result = _original_init_torch_distributed(self, *args, **kwargs)
    finally:
        # Restore the original function
        dist.init_process_group = original_init_pg
    return result


# Override the ModelRunner.init_torch_distributed method.
_OriginalModelRunner.init_torch_distributed = patched_init_torch_distributed
