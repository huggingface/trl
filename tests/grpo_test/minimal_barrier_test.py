# minimal_barrier_test.py
import os
import torch
import torch.distributed as dist
from accelerate import Accelerator


def main():
    accelerator = Accelerator()
    # Log basic distributed info
    print(
        f"[DEBUG] LOCAL_RANK: {os.environ.get('LOCAL_RANK')}, RANK: {os.environ.get('RANK')}, WORLD_SIZE: {os.environ.get('WORLD_SIZE')}"
    )
    print(
        f"[DEBUG] torch.distributed.is_initialized(): {torch.distributed.is_initialized()}"
    )

    # Set explicit device
    local_device = torch.device(f"cuda:{accelerator.process_index}")
    torch.cuda.set_device(local_device)
    print(
        f"[DEBUG] Process rank {accelerator.process_index} set to device {local_device}"
    )

    # Call an explicit torch.distributed.barrier() first
    current_device = torch.cuda.current_device()
    print(
        f"[DEBUG] Process rank {accelerator.process_index} calling torch.distributed.barrier() on device {current_device}"
    )
    torch.distributed.barrier(device_ids=[current_device])
    print(
        f"[DEBUG] Process rank {accelerator.process_index} passed torch.distributed.barrier()"
    )

    # Now call accelerator.wait_for_everyone()
    print(
        f"[DEBUG] Process rank {accelerator.process_index} calling accelerator.wait_for_everyone()"
    )
    accelerator.wait_for_everyone()
    print(
        f"[DEBUG] Process rank {accelerator.process_index} passed accelerator.wait_for_everyone()"
    )


if __name__ == "__main__":
    main()
