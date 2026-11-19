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

"""Train or evaluate a fixed Pi team on pinned AgentEscape Harbor tasks."""

import argparse
import asyncio
import importlib.metadata
import json
import random
import re
import time
from dataclasses import asdict
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from runtime import Settings, TeamRunner


REPO = "HuggingFaceSwarms/AgentEscapeBench-harbor"
REVISION = "144b647af561e28d681552cb04bc7b4a8dbeb1ce"


def split_tasks(names):
    names = sorted(names)
    if len(names) != 60:
        raise ValueError(f"Expected 60 tasks per difficulty at the pinned revision, found {len(names)}")
    random.Random(42).shuffle(names)
    return {"train": names[12:], "eval": names[:12]}


def split_difficulties(names, difficulties):
    names = list(names)
    splits = [
        split_tasks([name for name in names if name.startswith(f"agentescape-{level}-")]) for level in difficulties
    ]
    return {
        part: [name for pair in zip(*(split[part] for split in splits), strict=True) for name in pair]
        for part in ("train", "eval")
    }


def resolve_tasks(args):
    entries = HfApi().list_repo_tree(args.repo, "tasks", repo_type="dataset", revision=args.revision)
    names = [entry.path.removeprefix("tasks/") for entry in entries]
    splits = split_difficulties(names, args.difficulties)
    selected = splits["train" if args.command == "train" else "eval"]
    if args.num_tasks:
        selected = selected[: args.num_tasks]
    root = snapshot_download(
        args.repo,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.output_dir / "tasks-snapshot",
        allow_patterns=[f"tasks/{name}/**" for name in selected],
    )
    return [Path(root) / "tasks" / name for name in selected], splits


async def evaluate(settings, tasks, model, concurrency):
    runner = TeamRunner(settings)
    semaphore = asyncio.Semaphore(concurrency)

    async def one(task):
        async with semaphore:
            state, _ = await runner.run(task, model)
            return state

    await runner.start()
    try:
        results = await asyncio.gather(*(one(task) for task in tasks))
    finally:
        await runner.close()
    scored = [result["reward"] for result in results if result["reward"] is not None]
    summary = {
        "teams": len(results),
        "scored": len(scored),
        "unscorable": len(results) - len(scored),
        "success_rate": sum(scored) / len(scored) if scored else None,
    }
    (Path(settings.output_dir) / "evaluation.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "evaluate"])
    parser.add_argument("--vllm-url", required=True)
    parser.add_argument("--capture-url", required=True, help="Container-reachable URL of this host's capture port")
    parser.add_argument("--capture-port", type=int, default=8300)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--served-model", help="Named adapter to evaluate; defaults to --model")
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--backend", choices=["podman", "docker"], default="podman")
    parser.add_argument("--difficulties", type=int, nargs="+", default=[5])
    parser.add_argument("--full-finetune", action="store_true")
    parser.add_argument("--gateway-url")
    parser.add_argument("--max-staleness", type=int, default=2)
    parser.add_argument("--admission-deadline", type=float, default=0)
    parser.add_argument("--collection-deadline", type=float, default=0)
    parser.add_argument("--model-revision")
    parser.add_argument("--require-nonzero-update", action="store_true")
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Engineering check with synthetic rewards; not benchmark results",
    )
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--concurrent-teams", type=int, default=2)
    parser.add_argument("--teams-per-update", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--max-groups", type=int, default=32)
    parser.add_argument(
        "--num-tasks", type=int, default=0, help="0 selects the entire split; smoke: train 4, evaluate 2"
    )
    parser.add_argument("--max-turns", type=int, default=16)
    parser.add_argument("--agent-timeout", type=float, default=300)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--max-context", type=int, default=16384)
    parser.add_argument("--token-budget", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--report-to", choices=["none", "trackio"], default="none")
    parser.add_argument("--run-name")
    parser.add_argument("--trackio-project", default="trl-harbor-swarm")
    parser.add_argument("--trackio-space-id")
    parser.add_argument(
        "--time-limit-seconds", type=float, default=0, help="Stop and save after this training time; 0 disables"
    )
    args = parser.parse_args()
    for name in (
        "agents",
        "num_generations",
        "concurrent_teams",
        "teams_per_update",
        "max_steps",
        "max_turns",
        "max_output_tokens",
        "max_context",
        "token_budget",
    ):
        if vars(args)[name] <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_generations < 2 or args.temperature <= 0 or args.agent_timeout <= 0:
        parser.error("Require at least two generations, positive temperature, and positive agent timeout")
    if args.token_budget < args.max_context or args.max_output_tokens >= args.max_context:
        parser.error("Require token-budget >= max-context > max-output-tokens")
    if args.num_tasks < 0 or args.max_groups < 0 or args.time_limit_seconds < 0:
        parser.error("num-tasks, max-groups, and time-limit-seconds cannot be negative")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
        parser.error("--revision must be a full Hub commit SHA")
    if args.full_finetune and not args.gateway_url:
        parser.error("--full-finetune requires --gateway-url")
    if args.max_staleness < 0 or len(set(args.difficulties)) != len(args.difficulties):
        parser.error("Staleness must be nonnegative and difficulties must be unique")
    from accelerate import PartialState

    distributed = PartialState()
    if args.teams_per_update % distributed.num_processes:
        parser.error("--teams-per-update must be divisible by the training world size")
    with distributed.main_process_first():
        if distributed.is_main_process:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            if (args.output_dir / "manifest.json").exists():
                parser.error("Use a fresh output directory; resuming swarm rollouts is not supported")
            tasks, splits = resolve_tasks(args)
            (args.output_dir / "task-selection.json").write_text(
                json.dumps({"splits": splits, "tasks": [str(task.resolve()) for task in tasks]})
            )
        else:
            selection = json.loads((args.output_dir / "task-selection.json").read_text())
            tasks, splits = [Path(task) for task in selection["tasks"]], selection["splits"]
    settings = Settings(
        vllm_url=args.gateway_url or args.vllm_url,
        gateway_url=args.gateway_url,
        admission_deadline=args.admission_deadline,
        capture_url=args.capture_url,
        output_dir=str(args.output_dir.resolve()),
        model=args.model,
        backend=args.backend,
        agents=args.agents,
        capture_port=args.capture_port,
        max_turns=args.max_turns,
        agent_timeout=args.agent_timeout,
        max_output_tokens=args.max_output_tokens,
        max_context=args.max_context,
        temperature=args.temperature,
    )
    manifest = {
        "settings": asdict(settings),
        "repo": args.repo,
        "revision": args.revision,
        "splits": splits,
        "selected": [task.name for task in tasks],
        "command": args.command,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "versions": {name: importlib.metadata.version(name) for name in ("trl", "harbor", "openenv", "msgboard")},
    }
    if distributed.is_main_process:
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if args.command == "evaluate":
        asyncio.run(evaluate(settings, tasks, args.served_model or args.model, args.concurrent_teams))
        return
    from datasets import Dataset
    from peft import LoraConfig
    from training import SwarmConfig, SwarmTrainer
    from transformers import AutoTokenizer

    config = SwarmConfig(
        output_dir=str(args.output_dir / "checkpoint"),
        swarm_settings=asdict(settings),
        max_groups=args.max_groups,
        num_generations=args.num_generations,
        max_concurrent_rollouts=args.concurrent_teams,
        max_completion_length=args.max_output_tokens,
        token_budget=args.token_budget,
        gradient_accumulation_steps=args.teams_per_update // distributed.num_processes,
        collection_deadline=args.collection_deadline,
        ddp_timeout=7200,
        fsdp="full_shard auto_wrap" if distributed.num_processes > 1 else "",
        fsdp_config={
            "fsdp_version": 2,
            "cpu_ram_efficient_loading": True,
            "reshard_after_forward": True,
            "state_dict_type": "FULL_STATE_DICT",
        },
        model_init_kwargs={"revision": args.model_revision} if args.model_revision else None,
        max_steps=args.max_steps,
        max_staleness=args.max_staleness,
        score_centering=False,
        temperature=args.temperature,
        top_p=1.0,
        top_k=-1,
        chat_template_kwargs={"enable_thinking": False},
        epsilon=1.0,
        epsilon_high=1e9,
        vllm_importance_sampling_clip_min=0.2,
        vllm_importance_sampling_clip_max=5.0,
        vllm_server_base_url=args.vllm_url,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",
        dtype="bfloat16",
        bf16=True,
        activation_checkpointing=True,
        activation_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        report_to=args.report_to,
        run_name=args.run_name,
        project=args.trackio_project,
        trackio_space_id=args.trackio_space_id,
        trackio_static_space_id=False,
        hub_private_repo=True,
        log_completions=False,
        logging_steps=1,
        save_strategy="steps",
        save_steps=1,
        save_total_limit=3,
        seed=42,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    if args.synthetic_smoke:
        from smoke_full import SyntheticTrainer

        trainer_class = SyntheticTrainer
    else:
        trainer_class = SwarmTrainer
    trainer = trainer_class(
        model=args.model,
        args=config,
        processing_class=tokenizer,
        peft_config=None
        if args.full_finetune
        else LoraConfig(r=32, lora_alpha=64, target_modules="all-linear", exclude_modules=r".*visual.*"),
        train_dataset=Dataset.from_list([{"prompt": [{"role": "user", "content": str(task)}]} for task in tasks]),
    )
    import torch
    from transformers import TrainerCallback

    class ParameterProbe(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            self.initial = self.sample()

        def sample(self):
            return {
                name: (parameter.to_local() if hasattr(parameter, "to_local") else parameter)
                .detach()
                .reshape(-1)[:64]
                .cpu()
                .clone()
                for name, parameter in trainer.model.named_parameters()
                if parameter.requires_grad
            }

    probe = ParameterProbe()
    trainer.add_callback(probe)

    class CollectionLimit(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            if args.collection_deadline and time.time() >= args.collection_deadline:
                control.should_training_stop = True
                control.should_save = True

    trainer.add_callback(CollectionLimit())
    if args.time_limit_seconds:
        from transformers import TrainerCallback

        class TimeLimit(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                self.deadline = time.monotonic() + args_time_limit

            def on_step_end(self, args, state, control, **kwargs):
                if time.monotonic() >= self.deadline:
                    control.should_training_stop = True
                    control.should_save = True

        args_time_limit = args.time_limit_seconds
        trainer.add_callback(TimeLimit())
    trainer.train()
    trainer.save_model()
    if distributed.is_main_process:
        tokenizer.save_pretrained(config.output_dir)
    groups = [json.loads(line) for line in (args.output_dir / "groups.jsonl").read_text().splitlines()]
    changed = any(not torch.equal(probe.initial[name], parameter) for name, parameter in probe.sample().items())
    changed_tensor = torch.tensor(int(changed), device=distributed.device)
    if distributed.num_processes > 1:
        torch.distributed.all_reduce(changed_tensor, op=torch.distributed.ReduceOp.MAX)
    changed = bool(changed_tensor.item())
    result = {
        "optimizer_steps": trainer.state.global_step,
        "mixed_groups": sum(group["mixed"] for group in groups),
        "parameters_changed": changed,
    }
    if distributed.is_main_process:
        (args.output_dir / "training-smoke.json").write_text(json.dumps(result, indent=2))
    if args.require_nonzero_update and (not result["mixed_groups"] or not changed):
        raise RuntimeError("Training pipeline completed, but no mixed-reward group demonstrated a nonzero update")


if __name__ == "__main__":
    main()
