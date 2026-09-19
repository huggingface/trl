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

"""
Expert-parallel token dispatch on transformers main, installed at runtime.

transformers main shards a model at load time with `DistributedConfig`: FSDP2, tensor parallelism, and all-reduce
expert parallelism where the experts are split over the TP ranks. Token dispatch, where `ep_size` is independent of
`tp_size` and every rank trains on its own slice of the batch, is in huggingface/transformers#48792 and not merged.
This module carries that branch's changes, the Trainer fix for the parameters FSDP2 does not shard (PEFT adapters),
and the sharded-loading prefetch of #48227, and installs them onto the imported `transformers`.

    from trl.experimental.transformers_ep import DistributedConfig, patch_transformers

    patch_transformers()
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-30B-A3B-Instruct-2507", distributed_config=DistributedConfig(fsdp_size=8, ep_size=8)
    )

Written against transformers main at c587bc884d (5.18.0.dev0). Every patched target is looked up by attribute
first, so a rename on main fails at `patch_transformers()` rather than at the first forward. Supports `tp_size=1`
with dispatch; the Trainer's TP loss divisor from the branch is not carried.
"""

import os
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from fnmatch import fnmatchcase

import torch
import torch.distributed as dist
import transformers
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from transformers import PreTrainedModel, Trainer
from transformers.distributed import configuration_utils as _config_mod
from transformers.distributed import fsdp as _fsdp_mod
from transformers.distributed import mixin as _mixin_mod
from transformers.distributed import tensor_parallel as _tp_mod
from transformers.distributed import utils as _utils_mod
from transformers.distributed.configuration_utils import DistributedConfig as _BaseDistributedConfig
from transformers.utils import is_torch_greater_or_equal, logging


logger = logging.get_logger(__name__)

# Fetched up front so a rename on main is a clear AttributeError here, not a silent no-op later.
ALL_PARALLEL_STYLES = _tp_mod.ALL_PARALLEL_STYLES
_get_parameter_plan = _tp_mod._get_parameter_tp_plan
_AllReduceForward = _tp_mod._AllReduceForward
_AllReduceBackward = _tp_mod._AllReduceBackward
MoeExpertsParallel = _tp_mod.MoeExpertsParallel
apply_pipeline_parallelism = _mixin_mod.apply_pipeline_parallelism
_ensure_torch_distributed = _utils_mod._ensure_torch_distributed
_distributed_barrier = _utils_mod._distributed_barrier
_is_torch_distributed_initialized = _utils_mod._is_torch_distributed_initialized
fully_shard = _fsdp_mod.fully_shard
_get_fsdp_policy_kwargs = _fsdp_mod._get_fsdp_policy_kwargs
_resolve_tied_embed_lm_head_plan = _fsdp_mod._resolve_tied_embed_lm_head_plan
expand_fsdp_plan = _fsdp_mod.expand_fsdp_plan
is_norm_and_head_pair = _fsdp_mod.is_norm_and_head_pair


# ---------------------------------------------------------------------------------------------------------------
# DistributedConfig with `ep_size` and `ep_plan`
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class DistributedConfig(_BaseDistributedConfig):
    """
    [`~transformers.distributed.DistributedConfig`] plus expert parallelism decoupled from tensor parallelism.

    Args:
        ep_size (`int`, *optional*):
            Number of devices owning distinct expert shards. Defaults to 1. With token dispatch, must be a
            multiple of `tp_size` and divide `fsdp_size * tp_size`.
        ep_plan (`dict[str, str]`, *optional*):
            Expert parallel sharding plan. Leave as `None` to use the model's `base_model_ep_plan`, or pass a
            dictionary to override individual rules in it.
    """

    ep_size: int | None = None
    ep_plan: dict[str, str] | None = None

    @property
    def efsdp_size(self) -> int:
        """Size of the expert FSDP axis; the expert view is unused when EP is disabled."""
        return self.fsdp_size * self.tp_size // self.ep_size

    def __post_init__(self):
        self._resolve_parallelism()
        self._validate_mesh_config()

    def _resolve_parallelism(self):
        for value in (self.tp_size, self.fsdp_size, self.pp_size, self.ep_size):
            if value is not None and value < 1:
                raise ValueError(f"Parallelism sizes must be >= 1, got {value}.")

        if self.fsdp_size is None:
            self.fsdp_size = 1
        if self.pp_size is None:
            self.pp_size = 1
        if self.tp_size is None and self.tp_plan is not None:
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            other_parallel_size = self.fsdp_size * self.pp_size
            if world_size % other_parallel_size != 0:
                raise ValueError(
                    f"WORLD_SIZE ({world_size}) must be divisible by fsdp_size * pp_size "
                    f"({other_parallel_size}) to derive tp_size."
                )
            self.tp_size = world_size // other_parallel_size
        elif self.tp_size is None:
            self.tp_size = 1

        if self.enable_expert_parallel and self.ep_size is None:
            self.ep_size = self.tp_size
            warnings.warn(
                f"`enable_expert_parallel` without `ep_size` is deprecated. Use ep_size={self.ep_size} instead.",
                FutureWarning,
                stacklevel=4,
            )
        if self.ep_size is None:
            self.ep_size = 1
        # `PreTrainedModel.tp_plan` on main returns the expert plan when this is set. The patched property below
        # ignores it, so it only remains as a legacy attribute for callers.
        self.enable_expert_parallel = self.ep_size > 1

    def _validate_mesh_config(self):
        if self.ep_plan is not None and not isinstance(self.ep_plan, dict):
            raise ValueError("`ep_plan` must be a dictionary or None.")
        if self.ep_size > 1:
            if self.ep_size % self.tp_size:
                raise ValueError("`ep_size` must be a multiple of `tp_size` for token dispatch.")
            if (self.fsdp_size * self.tp_size) % self.ep_size:
                raise ValueError("`ep_size` must divide `fsdp_size * tp_size` for token dispatch.")
        if self.fsdp_size > 1 and self.pp_size > 1:
            raise ValueError(
                "Combining FSDP with pipeline parallelism is not supported yet. "
                "Use DistributedConfig(tp_size=N, fsdp_size=M), or combine TP and PP."
            )

    def _validate_resolved_ep_plan(self, ep_plan: dict[str, str]):
        if self.ep_size <= 1:
            return
        styles = set(ep_plan.values())
        if "ep_dispatch_experts" in styles:
            if self.pp_size > 1:
                raise ValueError("Combining token dispatch with pipeline parallelism is not supported yet.")
            if not is_torch_greater_or_equal("2.7"):
                raise OSError("Expert-parallel token dispatch requires `torch>=2.7`.")
        elif "ep_router" in styles and "moe_tp_experts" in styles:
            if self.ep_size != self.tp_size:
                raise ValueError("All-reduce EP requires `ep_size=tp_size` and identical tokens per EP group.")
        else:
            raise ValueError(
                "Invalid expert plan. Must contain either 'ep_dispatch_experts' for all-to-all dispatch "
                "or 'ep_router' and 'moe_tp_experts' for all-reduce masked EP."
            )


# ---------------------------------------------------------------------------------------------------------------
# Dense and expert meshes
# ---------------------------------------------------------------------------------------------------------------


class MeshManager:
    """Named access to dense and expert parallel axes without exposing their view selection."""

    def __init__(self, dense_mesh: DeviceMesh, expert_mesh: DeviceMesh):
        self._dense_mesh = dense_mesh
        self._expert_mesh = expert_mesh

    def get_mesh(self, dims: str | tuple[str, ...]) -> DeviceMesh:
        """Select expert axes for `ep`/`efsdp`, otherwise dense axes; DeviceMesh handles slicing."""
        dims = (dims,) if isinstance(dims, str) else dims
        mesh = self._expert_mesh if "ep" in dims or "efsdp" in dims else self._dense_mesh
        return mesh[dims]


def initialize_distributed_mesh(
    distributed_config: DistributedConfig,
) -> tuple[torch.device | None, MeshManager | None]:
    """Build named dense `(pp, fsdp, tp)` and expert `(pp, efsdp, ep)` views, both with singleton dims."""
    mesh_shape = (distributed_config.pp_size, distributed_config.fsdp_size, distributed_config.tp_size)
    if mesh_shape == (1, 1, 1):
        return None, None

    device_type = torch._C._get_accelerator().type
    if distributed_config.tp_size > 1 and device_type == "mps":
        raise RuntimeError("Tensor parallelism is not supported on MPS devices.")
    _ensure_torch_distributed(device_type)
    world_size = dist.get_world_size()
    expected_world_size = distributed_config.pp_size * distributed_config.fsdp_size * distributed_config.tp_size
    if expected_world_size != world_size:
        raise RuntimeError(
            f"The parallel mesh requires {expected_world_size} processes, but world_size is {world_size}."
        )
    if device_type != "cpu":
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        getattr(torch, device_type).set_device(local_rank)
        device_map = torch.device(device_type, local_rank)
    else:
        device_map = torch.device(device_type)

    dense_mesh = dist.init_device_mesh(device_type, mesh_shape, mesh_dim_names=("pp", "fsdp", "tp"))
    expert_mesh = dist.init_device_mesh(
        device_type,
        (distributed_config.pp_size, distributed_config.efsdp_size, distributed_config.ep_size),
        mesh_dim_names=("pp", "efsdp", "ep"),
    )
    return device_map, MeshManager(dense_mesh, expert_mesh)


# ---------------------------------------------------------------------------------------------------------------
# Token dispatch
# ---------------------------------------------------------------------------------------------------------------


class _ScaleGrad(torch.autograd.Function):
    """Identity whose backward scales the gradient."""

    @staticmethod
    def forward(ctx, tensor, scale):
        ctx.scale = scale
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


def dispatch_experts_forward(
    experts_forward: Callable,
    num_local_experts: int,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    ep_group,
    ep_size: int,
    tp_size: int = 1,
) -> torch.Tensor:
    """
    Expert-parallel forward by token dispatch. Every rank routes its own token slice, sends each selected
    (token, expert) pair to the rank that owns the expert with an all-to-all, runs its local experts on what it
    receives, sends the results back and combines them with the routing weights. The expert gradients sum
    contributions from `ep_size / tp_size` batches and are scaled by `tp_size / ep_size` before the remaining
    expert-data-parallel reduction, matching the trunk's FSDP average.
    """
    from torch.distributed.nn.functional import all_to_all_single

    num_tokens, hidden_dim = hidden_states.shape
    num_top_k = top_k_index.size(-1)

    # Sorting the selected pairs by expert groups them by owner rank, since each rank owns a contiguous range of
    # experts. The split sizes are the one host sync of the layer.
    expert_ids = top_k_index.reshape(-1)
    order = torch.argsort(expert_ids)
    send_tokens = hidden_states[order // num_top_k]
    send_counts = torch.zeros(num_local_experts * ep_size, dtype=torch.long, device=hidden_states.device)
    send_counts = send_counts.scatter_add_(0, expert_ids, torch.ones_like(expert_ids)).view(ep_size, num_local_experts)
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts, group=ep_group)
    send_sizes, recv_sizes = torch.stack([send_counts.sum(dim=1), recv_counts.sum(dim=1)]).tolist()
    recv_tokens = all_to_all_single(
        send_tokens.new_empty(sum(recv_sizes), hidden_dim),
        send_tokens,
        output_split_sizes=recv_sizes,
        input_split_sizes=send_sizes,
        group=ep_group,
    )
    recv_expert_ids = torch.arange(num_local_experts, device=hidden_states.device).repeat(ep_size)
    recv_expert_ids = recv_expert_ids.repeat_interleave(recv_counts.reshape(-1), output_size=sum(recv_sizes))

    expert_gradient_scale = tp_size / ep_size
    recv_tokens = _ScaleGrad.apply(recv_tokens, 1.0 / expert_gradient_scale)
    unit_weights = torch.ones_like(recv_expert_ids, dtype=recv_tokens.dtype).unsqueeze(-1)
    expert_out = experts_forward(recv_tokens, recv_expert_ids.unsqueeze(-1), unit_weights)
    expert_out = _ScaleGrad.apply(expert_out, expert_gradient_scale)

    recv_out = all_to_all_single(
        expert_out.new_empty(send_tokens.size(0), hidden_dim),
        expert_out,
        output_split_sizes=send_sizes,
        input_split_sizes=recv_sizes,
        group=ep_group,
    )
    inverse_order = torch.empty_like(order)
    inverse_order[order] = torch.arange(order.numel(), device=order.device)
    combined = recv_out[inverse_order] * top_k_weights.reshape(-1, 1)
    return combined.view(num_tokens, num_top_k, hidden_dim).sum(dim=1).to(hidden_states.dtype)


class EpDispatchExpertsParallel(MoeExpertsParallel):
    """Dispatch disjoint TP token slices to the experts' owners, then replicate the combined output on TP."""

    def install_forward(self, module, ep_mesh, *, is_expert_parallel=False, tp_mesh=None):
        original_forward = module.forward
        ep_group, ep_size = ep_mesh.get_group(), ep_mesh.size()
        tp_size = tp_mesh.size() if tp_mesh is not None else 1

        def experts_forward(hidden_states, top_k_index, top_k_weights):
            output = original_forward(hidden_states, top_k_index, top_k_weights)
            if hidden_states.size(0) == 0 and torch.is_grad_enabled():
                # Eager experts may return disconnected zeros on an empty receiver. Keep both the reverse
                # all-to-all and the expert FSDP reductions in the backward graph on every rank.
                output = output + hidden_states
                for param in module.parameters():
                    if isinstance(param, DTensor):
                        param = param.to_local()
                    output = output + param.reshape(-1)[:0].sum()
            return output

        def tp_forward(hidden_states, top_k_index, top_k_weights):
            if isinstance(hidden_states, DTensor):
                hidden_states = hidden_states.to_local()
            if isinstance(top_k_weights, DTensor):
                top_k_weights = top_k_weights.to_local()
            num_tokens = hidden_states.size(0)
            if tp_size > 1:
                hidden_states = _AllReduceBackward.apply(hidden_states, tp_mesh.get_group())
                top_k_weights = _AllReduceBackward.apply(top_k_weights, tp_mesh.get_group())
                tp_rank = tp_mesh.get_local_rank()
                rows = slice(num_tokens * tp_rank // tp_size, num_tokens * (tp_rank + 1) // tp_size)
                hidden_states, top_k_index, top_k_weights = hidden_states[rows], top_k_index[rows], top_k_weights[rows]
            with self.context_around_forward(module, ep_mesh):
                output = dispatch_experts_forward(
                    experts_forward,
                    module.num_experts,
                    hidden_states,
                    top_k_index,
                    top_k_weights,
                    ep_group,
                    ep_size,
                    tp_size=tp_size,
                )
            if tp_size > 1:
                full_output = output.new_zeros(num_tokens, output.size(-1))
                full_output[rows] = output
                output = _AllReduceForward.apply(full_output, tp_mesh.get_group())
            return output

        module.forward = tp_forward
        return module


# ---------------------------------------------------------------------------------------------------------------
# Plan resolution and application
# ---------------------------------------------------------------------------------------------------------------


def _validate_parallel_plan_styles(plan: dict[str, str] | None) -> None:
    unsupported = {style for style in (plan or {}).values() if style not in ALL_PARALLEL_STYLES}
    if unsupported:
        raise ValueError(
            f"Unsupported parallel styles: {unsupported}. Supported styles are {list(ALL_PARALLEL_STYLES.keys())}"
        )


def resolve_parallel_plans(model: nn.Module, distributed_config: DistributedConfig):
    """Resolve overrides and give EP ownership of its modules before applying any sharding."""
    model_names = {name for name, _ in model.named_modules()} | {name for name, _ in model.named_parameters()}
    for plan_name in ("tp_plan", "ep_plan"):
        override = getattr(distributed_config, plan_name)
        if not isinstance(override, dict):
            continue
        valid_names = model_names | set(getattr(model, plan_name))
        for pattern in override:
            if not any(fnmatchcase(name, pattern) for name in valid_names):
                raise ValueError(
                    f"The `{plan_name}` pattern {pattern!r} does not match any module, parameter, or existing plan "
                    f"entry in {type(model).__name__}. Check the full path, including any 'model.' prefix."
                )

    if isinstance(distributed_config.tp_plan, dict):
        model.tp_plan = model.tp_plan | distributed_config.tp_plan
    if isinstance(distributed_config.ep_plan, dict):
        model.ep_plan = model.ep_plan | distributed_config.ep_plan

    tp_plan = model.tp_plan if distributed_config.tp_size > 1 else {}
    ep_plan = model.ep_plan if distributed_config.ep_size > 1 else {}

    if distributed_config.ep_size > 1 and not ep_plan:
        raise ValueError(
            f"{type(model).__name__} does not define an expert-parallel plan. Pass `ep_plan` in DistributedConfig, "
            "add `base_model_ep_plan` to the model's config, or disable expert parallelism."
        )
    distributed_config.ep_plan = dict(ep_plan)

    def is_expert_path(name):
        return any(fnmatchcase(name, path) or fnmatchcase(name, path + ".*") for path in expert_paths)

    expert_paths = list(ep_plan)
    if "ep_dispatch_experts" in ep_plan.values():
        expert_paths = [name for name, style in ep_plan.items() if style in ("moe_tp_experts", "ep_dispatch_experts")]
        ep_plan = {name: style for name, style in ep_plan.items() if is_expert_path(name)}
    tp_plan = {name: style for name, style in tp_plan.items() if not is_expert_path(name)}
    _validate_parallel_plan_styles(tp_plan)
    _validate_parallel_plan_styles(ep_plan)
    return tp_plan, ep_plan


def _apply_plan(model: nn.Module, shard_mesh: DeviceMesh, plan: dict[str, str], *, tp_mesh: DeviceMesh | None = None):
    """Shard the parameters a plan names on `shard_mesh` and install the styles' forward hooks."""
    for name, module in model.named_modules():
        for p_name, _ in list(module.named_parameters(recurse=False)):
            full = f"{name}.{p_name}" if name else p_name
            style_name = _get_parameter_plan(full, plan, is_weight=True)
            if style_name is not None and style_name in ALL_PARALLEL_STYLES:
                style = ALL_PARALLEL_STYLES[style_name]
                style.validate_param(module, p_name, shard_mesh, parameter_name=full)
                style.shard_param(module, p_name, shard_mesh)

        style_name = _get_parameter_plan(name, plan, is_weight=False)
        if style_name is not None and style_name in ALL_PARALLEL_STYLES:
            style = ALL_PARALLEL_STYLES[style_name]
            if style_name == "mla_kv_a_proj":
                module.config = model.config.get_text_config()
            if style_name == "ep_dispatch_experts":
                style.install_forward(module, ep_mesh=shard_mesh, tp_mesh=tp_mesh)
            else:
                style.install_forward(module, shard_mesh)
        module._is_hooked = True
    return model


def apply_fully_sharded_data_parallelism(model: nn.Module, mesh_manager: MeshManager) -> nn.Module:
    """
    FSDP2 over the dense `fsdp` axis. With token dispatch, routed experts are wrapped separately on `efsdp`,
    even when its size is one, so the surrounding wrapper excludes them.
    """
    distributed_config = model.config.distributed_config
    fsdp_mesh = mesh_manager.get_mesh("fsdp")
    fsdp_plan = dict(getattr(model, "_fsdp_plan", None) or {})
    if not fsdp_plan:
        raise ValueError(
            f"{type(model).__name__} does not have a FSDP2 plan declared. Set "
            "`base_model_fsdp_plan` on the config and `_fsdp_plan` on the head class."
        )
    adapted_fsdp_plan = _resolve_tied_embed_lm_head_plan(fsdp_plan, model)
    reshard_targets, no_reshard_targets = expand_fsdp_plan(model, adapted_fsdp_plan)

    fsdp_policy_kwargs = _get_fsdp_policy_kwargs(distributed_config)
    if distributed_config.ep_size > 1 and "ep_dispatch_experts" in model.ep_plan.values():
        expert_mesh = mesh_manager.get_mesh("efsdp")
        for module in model.modules():
            if getattr(module, "_is_expert_parallel", False):
                fully_shard(module, mesh=expert_mesh, reshard_after_forward=True, **fsdp_policy_kwargs)

    for _, module in reshard_targets:
        fully_shard(module, mesh=fsdp_mesh, reshard_after_forward=True, **fsdp_policy_kwargs)
    if is_norm_and_head_pair(no_reshard_targets, model):
        fully_shard(
            [m for _, m in no_reshard_targets], mesh=fsdp_mesh, reshard_after_forward=False, **fsdp_policy_kwargs
        )
    else:
        for _, module in no_reshard_targets:
            fully_shard(module, mesh=fsdp_mesh, reshard_after_forward=False, **fsdp_policy_kwargs)
    fully_shard(model, mesh=fsdp_mesh, **fsdp_policy_kwargs)
    model._is_fsdp_managed_module = True
    return model


# ---------------------------------------------------------------------------------------------------------------
# Replacements for the PreTrainedModel mixin
# ---------------------------------------------------------------------------------------------------------------


def _tp_plan_get(self):
    return self._tp_plan if self._tp_plan is not None else {}


def _ep_plan_get(self):
    return self._ep_plan if self._ep_plan is not None else {}


def _ep_plan_set(self, plan):
    plan = {} if plan is None else plan
    if not isinstance(plan, dict):
        raise ValueError("Can only set a dictionary as `ep_plan`")
    _validate_parallel_plan_styles(plan)
    self._ep_plan = plan


@classmethod
def _prepare_distribute_model(cls, distributed_config, device_map=None):
    if distributed_config is None:
        return None, device_map, None
    if isinstance(distributed_config, dict):
        distributed_config = DistributedConfig.from_dict(distributed_config)
    if not isinstance(distributed_config, DistributedConfig):
        distributed_config = DistributedConfig.from_dict(distributed_config.to_dict())
    if distributed_config.tp_size == 1 and distributed_config.fsdp_size == 1 and distributed_config.pp_size == 1:
        return distributed_config, device_map, None
    if distributed_config.tp_size > 1 and device_map is not None:
        raise ValueError("Tensor parallelism and `device_map` are mutually exclusive.")
    if distributed_config.fsdp_size > 1 and not is_torch_greater_or_equal("2.7"):
        raise OSError("FSDP2 requires `torch>=2.7` (distributed checkpoint save/load).")
    device_map, mesh_manager = initialize_distributed_mesh(distributed_config)
    # `from_pretrained` hands the third value to the loader as the dense mesh. The manager rides along on the
    # config for `maybe_distribute_model`, which is the only other place that needs the expert axes.
    distributed_config._mesh_manager = mesh_manager
    return distributed_config, device_map, mesh_manager.get_mesh(("pp", "fsdp", "tp"))


@classmethod
def _maybe_distribute_model(cls, model, distributed_config, device_mesh):
    mesh_manager = getattr(distributed_config, "_mesh_manager", None)
    if mesh_manager is None:
        return model

    model.config.distributed_config = distributed_config
    model._mesh_manager = mesh_manager
    model._device_mesh = device_mesh
    model._tp_size = distributed_config.tp_size
    model._fsdp_size = distributed_config.fsdp_size

    tp_plan, ep_plan = resolve_parallel_plans(model, distributed_config)
    distributed_config._validate_resolved_ep_plan(ep_plan)

    if distributed_config.pp_size > 1:
        model = apply_pipeline_parallelism(model, mesh_manager.get_mesh("pp"))
    if tp_plan:
        model = _apply_plan(model, mesh_manager.get_mesh("tp"), tp_plan)
    if ep_plan:
        if "ep_dispatch_experts" in ep_plan.values():
            model = _apply_plan(model, mesh_manager.get_mesh("ep"), ep_plan, tp_mesh=mesh_manager.get_mesh("tp"))
        else:
            model = _apply_plan(model, mesh_manager.get_mesh("tp"), ep_plan)
    if distributed_config.fsdp_size > 1 or "ep_dispatch_experts" in ep_plan.values():
        model = apply_fully_sharded_data_parallelism(model, mesh_manager)
    return model


# ---------------------------------------------------------------------------------------------------------------
# Sharded-loading prefetch (#48227)
# ---------------------------------------------------------------------------------------------------------------


def prefetch_checkpoint_shards(checkpoint_files: list[str]) -> None:
    """
    Warm the page cache for the checkpoint shards before the per-tensor loading pass, opt-in via
    `HF_SHARD_PREFETCH=<read threads per rank>`. Local ranks split the shard list between them.
    """
    prefetch_threads = int(os.environ.get("HF_SHARD_PREFETCH", "0"))
    if not checkpoint_files or not prefetch_threads:
        return
    from concurrent.futures import ThreadPoolExecutor

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))

    def _warm(path, bufsize=16 * 2**20):
        with open(path, "rb", buffering=0) as f:
            while f.read(bufsize):
                pass

    prefetch_start = time.time()
    with ThreadPoolExecutor(max_workers=prefetch_threads) as pool:
        list(pool.map(_warm, checkpoint_files[local_rank::local_world]))
    _distributed_barrier()
    logger.warning_once(f"Prefetched {len(checkpoint_files)} checkpoint shards in {time.time() - prefetch_start:.0f}s")


# ---------------------------------------------------------------------------------------------------------------
# Trainer: prepare a model sharded at load time, and keep unsharded parameters in sync
# ---------------------------------------------------------------------------------------------------------------


def _sync_unsharded_params(model: nn.Module) -> None:
    """
    Parameters added after `fully_shard`, such as PEFT adapters, are plain tensors replicated on every rank.
    They are initialised before the Trainer seeds the RNG, so ranks start from different values, and FSDP2
    reduce-scatters only what it sharded, so each rank would keep the gradient of its own batch. DDP did both jobs
    for replicated parameters; this path skips the DDP wrap, so do the same here.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return

    def average(param):
        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

    count = 0
    for param in model.parameters():
        if param.requires_grad and not isinstance(param, DTensor) and not isinstance(param.data, DTensor):
            dist.broadcast(param.data, src=0)
            param.register_post_accumulate_grad_hook(average)
            count += 1
    if count:
        logger.info(f"Synchronised {count} parameters that FSDP2 does not shard: rank 0's values, averaged gradients.")


def _make_prepare_for_training(original):
    def _prepare_for_training(self, max_steps, train_dataloader, resume_from_checkpoint):
        model = self.model
        if not (
            getattr(model, "_is_fsdp_managed_module", False)
            and not self.is_fsdp_enabled
            and not self.is_deepspeed_enabled
        ):
            return original(self, max_steps, train_dataloader, resume_from_checkpoint)

        # Sharded at load time: the model already owns placement and gradient reduction. Prepare autocast and
        # compilation only, without asking Accelerate to wrap the DTensor parameters in DDP or to shard them again.
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False
        self.create_optimizer()
        model = self.accelerator.prepare_model(model, device_placement=False, evaluation_mode=True)
        self.optimizer = self.accelerator.prepare(self.optimizer)
        _sync_unsharded_params(model)
        self.create_scheduler(num_training_steps=max_steps)
        self.model_wrapped = model
        return model, train_dataloader

    return _prepare_for_training


# ---------------------------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------------------------

_PATCHED = False


def patch_transformers() -> None:
    """Install expert-parallel token dispatch, the Trainer fix and the loading prefetch onto `transformers`."""
    global _PATCHED
    if _PATCHED:
        return

    ALL_PARALLEL_STYLES["ep_dispatch_experts"] = EpDispatchExpertsParallel()

    # Both names, so `from transformers.distributed import DistributedConfig` after the patch gets the new one.
    _config_mod.DistributedConfig = DistributedConfig
    transformers.distributed.DistributedConfig = DistributedConfig

    _utils_mod.MeshManager = MeshManager
    _utils_mod.initialize_distributed_mesh = initialize_distributed_mesh
    _mixin_mod.initialize_distributed_mesh = initialize_distributed_mesh

    PreTrainedModel.tp_plan = property(_tp_plan_get, PreTrainedModel.tp_plan.fset)
    PreTrainedModel.ep_plan = property(_ep_plan_get, _ep_plan_set)
    PreTrainedModel.prepare_distribute_model = _prepare_distribute_model
    PreTrainedModel.maybe_distribute_model = _maybe_distribute_model

    original_load = PreTrainedModel._load_pretrained_model

    @staticmethod
    def _load_pretrained_model(model, state_dict, checkpoint_files, load_config, expected_keys=None):
        prefetch_checkpoint_shards(checkpoint_files)
        return original_load(model, state_dict, checkpoint_files, load_config, expected_keys)

    PreTrainedModel._load_pretrained_model = _load_pretrained_model

    Trainer._prepare_for_training = _make_prepare_for_training(Trainer._prepare_for_training)

    # The only shipped MoE plan that names the experts for dispatch; the others keep all-reduce EP.
    from transformers import Qwen3MoeConfig

    Qwen3MoeConfig.base_model_ep_plan = {
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "ep_dispatch_experts",
    }

    _PATCHED = True
