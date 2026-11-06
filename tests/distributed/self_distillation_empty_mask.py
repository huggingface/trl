# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train with rank-local empty targets without replacing trainer internals or generation."""

import argparse
import json
import math
import os
from contextlib import nullcontext
from pathlib import Path

import torch
from datasets import Dataset
from transformers import TrainerCallback

from trl.experimental.sdft import SDFTConfig, SDFTTrainer
from trl.experimental.sdpo import SDPOConfig, SDPOTrainer


class MaskCaptureCallback(TrainerCallback):
    def __init__(self, trainer_name, empty):
        self.trainer_name = trainer_name
        self.empty = empty
        self.checked_batches = 0

    def on_teacher_context_built(self, self_distillation_mask, **kwargs):
        assert (self_distillation_mask == (0 if self.empty else 1)).all()
        self.checked_batches += 1

    def on_self_distillation_batch_prepared(self, completion_ids, **kwargs):
        if self.trainer_name == "sdft":
            # Two skipped tokens mask the short completion, but not the four-token completion.
            assert completion_ids.size(1) == (2 if self.empty else 4)
            self.checked_batches += 1


class FixedReward:
    def __init__(self, pattern, rank):
        self.pattern = pattern
        self.rank = rank

    def __call__(self, prompts, **kwargs):
        if self.pattern == "all_empty":
            return [0.0] * len(prompts)
        if self.pattern == "nonempty":
            return [1.0] * len(prompts)
        # Only the successful row has no other successful demonstration in its generation group.
        return [float(self.rank == int(self.pattern[-1]))] * len(prompts)


class ParameterCaptureCallback(TrainerCallback):
    def __init__(self, accelerator):
        self.accelerator = accelerator
        self.before = None
        self.after = None
        self.skipped_steps = []

    def on_train_begin(self, args, state, control, model, **kwargs):
        self.before = self._snapshot(model)

    def on_train_end(self, args, state, control, model, **kwargs):
        self.after = self._snapshot(model)

    def on_step_end(self, args, state, control, **kwargs):
        self.skipped_steps.append(self.accelerator.optimizer_step_was_skipped)

    def _snapshot(self, model):
        context = nullcontext()
        if self.accelerator.state.deepspeed_plugin is not None:
            from deepspeed import zero

            context = zero.GatheredParameters(model.parameters())
        with context:
            return {
                name: param.detach().cpu().clone() for name, param in model.named_parameters() if param.requires_grad
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", choices=["sdpo", "sdft"], required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--distillation_mode", default="sampled_token")
    parser.add_argument("--pattern", choices=["empty0", "empty1", "all_empty", "nonempty"], default="empty0")
    parser.add_argument("--precision", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    empty = args.pattern == "all_empty" or args.pattern == f"empty{rank}"
    callback = MaskCaptureCallback(args.trainer, empty)
    dataset = Dataset.from_dict({"prompt": ["Solve 2+2."] * 4, "privileged_context": ["The answer is 4."] * 4})
    config_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": 1 if int(os.environ.get("WORLD_SIZE", "1")) > 1 else 2,
        "gradient_accumulation_steps": 1,
        "num_generations": 2,
        "max_steps": 2,
        "max_completion_length": 4,
        "max_prompt_length": 32,
        "distillation_mode": args.distillation_mode,
        "distillation_alpha": 1.0,
        "distillation_topk": 5,
        "distillation_is_clip": None,
        "model_init_kwargs": {
            "dtype": "float32" if args.precision == "float16" else args.precision,
            "attn_implementation": "eager",
        },
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": "none",
        "disable_tqdm": True,
        "use_cpu": args.cpu,
        "bf16": args.precision == "bfloat16",
        "fp16": args.precision == "float16",
        "ddp_timeout": 60,
        "seed": 42,
        "generation_kwargs": {"min_new_tokens": 4, "max_new_tokens": 4},
    }
    trainer_kwargs = {
        "model": "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
        "train_dataset": dataset,
        "callbacks": [callback],
    }
    if args.trainer == "sdpo":
        config = SDPOConfig(**config_kwargs, success_reward_threshold=0.5, include_environment_feedback=False)
        trainer = SDPOTrainer(args=config, reward_funcs=FixedReward(args.pattern, rank), **trainer_kwargs)
    else:
        config_kwargs["generation_kwargs"] = {
            "min_new_tokens": 2 if empty else 4,
            "max_new_tokens": 2 if empty else 4,
        }
        config = SDFTConfig(**config_kwargs, num_loss_tokens_to_skip=2)
        trainer = SDFTTrainer(args=config, **trainer_kwargs)

    parameters = ParameterCaptureCallback(trainer.accelerator)
    trainer.add_callback(parameters)
    trainer.train()

    assert trainer.state.global_step == 2
    assert callback.checked_batches == 2
    losses = [row["self_distillation/distillation_loss"] for row in trainer.state.log_history if "grad_norm" in row]
    grad_norms = [float(row["grad_norm"]) for row in trainer.state.log_history if "grad_norm" in row]
    assert len(losses) == len(grad_norms) == 2
    assert all(math.isfinite(value) for value in losses + grad_norms)
    assert parameters.before.keys() == parameters.after.keys()
    assert all(torch.isfinite(param).all() for param in parameters.after.values())
    changed_parameters = [
        name for name in parameters.before if not torch.equal(parameters.before[name], parameters.after[name])
    ]
    (Path(args.output_dir) / f"parameter_checks_rank{rank}.json").write_text(
        json.dumps(
            {
                "trainer": args.trainer,
                "precision": args.precision,
                "pattern": args.pattern,
                "rank": rank,
                "losses": losses,
                "grad_norms": grad_norms,
                "skipped_steps": parameters.skipped_steps,
                "trainable_parameters": len(parameters.before),
                "changed_parameters": changed_parameters,
            },
            indent=2,
        )
        + "\n"
    )
    assert parameters.skipped_steps == [False, False]
    if args.pattern == "all_empty":
        assert losses == [0.0, 0.0]
        assert grad_norms == [0.0, 0.0]
        assert not changed_parameters
    else:
        assert any(value > 0 for value in grad_norms)
        assert changed_parameters
    gathered_losses = trainer.accelerator.gather(torch.tensor(losses, device=trainer.accelerator.device))
    torch.testing.assert_close(
        gathered_losses.reshape(-1, 2), gathered_losses[:2].expand_as(gathered_losses.reshape(-1, 2))
    )
    trainer.accelerator.wait_for_everyone()
    trainer.accelerator.end_training()


if __name__ == "__main__":
    main()
