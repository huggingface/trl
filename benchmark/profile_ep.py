"""Profile SFT training with Expert Parallel using torch.profiler.

Reuses the full SFTTrainer EP init path (device mesh, DistributedConfig, FSDP2)
and injects a ProfilerCallback to capture traces viewable in Perfetto / TensorBoard.

Usage (single node, 8 GPUs):
    torchrun --nproc_per_node=8 benchmark/profile_ep.py

The trace files land in benchmark/profiler_logs/ep_30b/ (one .json.gz per rank).
View with:
    - Drag into https://ui.perfetto.dev/
    - Or: pip install torch_tb_profiler && tensorboard --logdir benchmark/profiler_logs/ep_30b
"""

import torch


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from datasets import load_dataset
from torch.profiler import ProfilerActivity, schedule, tensorboard_trace_handler
from transformers.trainer_callback import TrainerCallback

from trl import SFTConfig, SFTTrainer


# -- Profiler callback --------------------------------------------------------

WAIT = 1
WARMUP = 1
ACTIVE = 3
TOTAL_STEPS = WAIT + WARMUP + ACTIVE  # 5


class ProfilerCallback(TrainerCallback):
    """Wraps torch.profiler.profile around the training loop."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.prof = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.prof = torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=WAIT, warmup=WARMUP, active=ACTIVE, repeat=1),
            on_trace_ready=tensorboard_trace_handler(self.log_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        self.prof.start()

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()
        if state.global_step >= TOTAL_STEPS:
            control.should_training_stop = True

    def on_train_end(self, args, state, control, **kwargs):
        self.prof.stop()


# -- Main ---------------------------------------------------------------------

LOG_DIR = "benchmark/profiler_logs/ep_30b"
MODEL = "Qwen/Qwen3-30B-A3B"
DATASET = "THUDM/LongAlign-10k"
CTX_LEN = 8192


def main():
    dataset = load_dataset(DATASET, split="train")

    training_args = SFTConfig(
        output_dir="/tmp/profile_ep_30b",
        max_steps=TOTAL_STEPS + 1,  # +1 so trainer doesn't stop before callback
        per_device_train_batch_size=1,
        gradient_checkpointing=True,
        bf16=True,
        max_length=CTX_LEN,
        packing=True,
        packing_strategy="wrapped",
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        enable_expert_parallel=True,
        include_num_input_tokens_seen=True,
        model_init_kwargs={"attn_implementation": "sdpa"},
    )

    trainer = SFTTrainer(
        model=MODEL,
        args=training_args,
        train_dataset=dataset,
        callbacks=[ProfilerCallback(LOG_DIR)],
    )

    trainer.train()

    if torch.distributed.get_rank() == 0:
        print(f"\nProfiler traces saved to {LOG_DIR}/")
        print("View with: https://ui.perfetto.dev/ (drag the .json.gz file)")


if __name__ == "__main__":
    main()
