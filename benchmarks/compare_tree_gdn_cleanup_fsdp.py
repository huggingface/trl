"""Exact pre/post-cleanup Qwen FSDP2 regression, including checkpointing and SGD.

Run with torchrun (one or two GPUs). This compares tree packing to itself before
cleanup, NOT to ordinary packing; the separate ordinary-packing failures remain.
"""

import argparse
import copy
import json
import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers
from benchmarks.compare_tree_gdn_cleanup import load_before, matching_configs
from packaging.version import Version
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from trl.experimental.async_grpo.tree import (
    PrefixForest,
    build_tree_block_mask,
    dfs_intervals,
    register_tree_attention,
)
from trl.experimental.async_grpo.tree.gdn import enable_qwen3_5_tree_gdn
from trl.kernels._tree_gdn_fla import tree as current
from trl.kernels.tree_gated_delta_rule import TreeGDNPlan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank = dist.get_rank()
    torch.manual_seed(42)
    register_tree_attention()
    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        dtype=torch.bfloat16,
        use_cache=False,
        attn_implementation="sdpa",
    )
    with torch.device("cuda"):
        model = Qwen3_5ForCausalLM(config).to(torch.bfloat16).train()
    if Version(transformers.__version__) == Version("5.8.1"):
        # Test-only alias for the installed version; the adapter uses the newer name.
        for layer in model.model.layers:
            layer.block_type = layer.layer_type
    model.set_attn_implementation("tree_flex")
    enable_qwen3_5_tree_gdn(model)
    baseline = copy.deepcopy(model)
    for network in (model, baseline):
        network.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for layer in network.model.layers:
            if layer.block_type == "linear_attention":
                fully_shard(layer.linear_attn, mp_policy=policy)
            fully_shard(layer, mp_policy=policy)
        fully_shard(network, mp_policy=policy)
    optimizers = [torch.optim.SGD(network.parameters(), lr=0.01) for network in (baseline, model)]
    current_function = current.TreeGatedDeltaRule
    report = dict(before=args.before, world_size=dist.get_world_size(), rank=rank, checkpointing=True, steps=[])
    with tempfile.TemporaryDirectory(prefix="tree-gdn-fsdp-before-") as directory:
        before = load_before(args.before, directory)
        matching_configs(before)
        for step in range(2):
            prefix = list(range(1, 9 + rank * 3 + step))
            rows = [prefix + [20, 21, 22], prefix + [25, 26 + step]]
            if rank == 1:
                rows.append(prefix + [20, 21, 28, 29])
            forest = PrefixForest()
            for row in rows:
                forest.insert(row, 0)
            packed, layout, mapping = forest.linearize()
            plan = TreeGDNPlan.build(layout.offsets, layout.parents, "cuda")
            enter, leave = [torch.tensor(t, device="cuda") for t in dfs_intervals(layout)]
            mask = build_tree_block_mask(enter, leave)
            losses = []
            for network, optimizer, function in zip(
                (baseline, model), optimizers, (before.TreeGatedDeltaRule, current_function), strict=True
            ):
                # Keep this binding through backward: checkpointing recomputes forward.
                current.TreeGatedDeltaRule = function
                optimizer.zero_grad(set_to_none=True)
                output = network(
                    torch.tensor([packed], device="cuda"),
                    position_ids=layout.position_ids.cuda()[None],
                    tree_block_mask=mask,
                    tree_gdn_plan=plan,
                    use_cache=False,
                ).logits
                loss = sum(
                    F.cross_entropy(
                        output[:, [mapping[t] for t in forest.walk(row, 0)][:-1]].float().squeeze(0),
                        torch.tensor(row[1:], device="cuda"),
                    )
                    for row in rows
                )
                loss.backward()
                losses.append(loss.detach())
            torch.testing.assert_close(*losses, atol=0, rtol=0)
            for (name, actual), (_, expected) in zip(
                model.named_parameters(), baseline.named_parameters(), strict=True
            ):
                torch.testing.assert_close(
                    actual.grad.full_tensor(), expected.grad.full_tensor(), atol=0, rtol=0, msg=name
                )
            for optimizer in optimizers:
                optimizer.step()
            for (name, actual), (_, expected) in zip(
                model.named_parameters(), baseline.named_parameters(), strict=True
            ):
                torch.testing.assert_close(actual.full_tensor(), expected.full_tensor(), atol=0, rtol=0, msg=name)
            report["steps"].append(dict(step=step, loss=losses[1].item(), exact_gradients=True, exact_weights=True))
            args.output.with_name(f"{args.output.stem}-rank{rank}.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(  # noqa: T201
                f"rank={rank} step={step}: loss, every parameter gradient, and updated weights match exactly",
                flush=True,
            )
    current.TreeGatedDeltaRule = current_function
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
