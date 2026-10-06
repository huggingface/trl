"""Two-rank loss, gradient, and optimizer parity; run with torchrun --nproc-per-node=2.

The reference duplicates rollouts and explicitly averages unsharded gradients across ranks. The tree model shards both
decoder blocks and GDN blocks, exercising their forward hooks.
"""

import argparse
import copy
import os
from contextlib import nullcontext
from functools import partial

import torch
import torch.distributed as dist
import torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision, MixedPrecisionPolicy, fully_shard
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5ForCausalLM,
    Qwen3_5GatedDeltaNet,
)

from trl.experimental.async_grpo.tree import (
    PrefixForest,
    build_tree_block_mask,
    dfs_intervals,
    register_tree_attention,
)
from trl.experimental.async_grpo.tree.gdn import enable_qwen3_5_tree_gdn
from trl.kernels.tree_gated_delta_rule import TreeGDNPlan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsdp-version", type=int, choices=[1, 2], default=2)
    parser.add_argument("--checkpointing", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
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
        baseline = Qwen3_5ForCausalLM(config).to(torch.bfloat16).train()
    for layer in baseline.model.layers:
        if layer.block_type == "linear_attention":
            layer.linear_attn.chunk_gated_delta_rule = chunk_gated_delta_rule
    model = copy.deepcopy(baseline)
    model.set_attn_implementation("tree_flex")
    enable_qwen3_5_tree_gdn(model)
    if args.checkpointing:
        for m in (model, baseline):
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.fsdp_version == 2:
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for layer in model.model.layers:
            if layer.block_type == "linear_attention":
                fully_shard(layer.linear_attn, mp_policy=policy)
            fully_shard(layer, mp_policy=policy)
        fully_shard(model, mp_policy=policy)
    else:
        model = FullyShardedDataParallel(
            model,
            use_orig_params=True,
            device_id=torch.cuda.current_device(),
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
            auto_wrap_policy=partial(
                transformer_auto_wrap_policy, transformer_layer_cls={Qwen3_5DecoderLayer, Qwen3_5GatedDeltaNet}
            ),
        )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    reference_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.01)
    for step in range(2):
        # Different topology on each rank and each step; no state or layout may leak between forwards.
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
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        output = model(
            torch.tensor([packed], device="cuda"),
            position_ids=layout.position_ids.cuda()[None],
            tree_block_mask=build_tree_block_mask(enter, leave),
            tree_gdn_plan=plan,
            use_cache=False,
        ).logits
        loss, ref_loss = 0, 0
        for row in rows:
            index = [mapping[t] for t in forest.walk(row, 0)]
            ref = baseline(torch.tensor([row], device="cuda"), use_cache=False).logits
            target = torch.tensor(row[1:], device="cuda")
            loss = loss + F.cross_entropy(output[:, index[:-1]].float().squeeze(0), target)
            ref_loss = ref_loss + F.cross_entropy(ref[:, :-1].float().squeeze(0), target)
        torch.testing.assert_close(loss, ref_loss, atol=0.02, rtol=0.003)
        loss.backward()
        ref_loss.backward()
        for p in baseline.parameters():
            grad = p.grad.float()
            dist.all_reduce(grad)
            p.grad.copy_(grad / dist.get_world_size())
        context = (
            FullyShardedDataParallel.summon_full_params(model, with_grads=True, writeback=False)
            if args.fsdp_version == 1
            else nullcontext()
        )
        max_error = 0
        with context:
            for (name, actual), (_, expected) in zip(
                model.named_parameters(), baseline.named_parameters(), strict=True
            ):
                assert actual.grad is not None, name
                grad = actual.grad.full_tensor() if args.fsdp_version == 2 else actual.grad
                relative_rms = (grad.float() - expected.grad.float()).norm() / expected.grad.float().norm().clamp_min(
                    1e-8
                )
                assert relative_rms < 0.06, (name, relative_rms.item())
                max_error = max(max_error, relative_rms.item())
        optimizer.step()
        reference_optimizer.step()
        print(  # noqa: T201
            f"rank={rank} fsdp={args.fsdp_version} checkpoint={args.checkpointing} step={step} "
            f"loss={loss.item():.6f} max_gradient_relative_rms={max_error:.6f}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
