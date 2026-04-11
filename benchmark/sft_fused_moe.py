# ruff: noqa: T201
"""
SFT training script for MoE models with fused expert weights.

Fuses individual expert nn.Linear modules into grouped tensors so FSDP2
can shard them symmetrically across ranks (fixing the collective mismatch
that prevents vanilla FSDP2 from training MoE models).

Usage:
    accelerate launch --config_file <fsdp2_config.yaml> benchmark/sft_fused_moe.py \
        --model_name_or_path Qwen/Qwen3-30B-A3B \
        --dataset_name THUDM/LongAlign-10k \
        --max_length 16384 \
        ...
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


def fuse_moe_experts(model):
    """
    Fuse individual MoE expert weights into grouped tensors.

    Replaces ModuleList[MLP(gate_proj, up_proj, down_proj)] with a single Module
    holding fused Parameter tensors of shape [num_experts, ...], and patches the
    forward method to use indexed F.linear calls.

    This makes expert weights symmetric across FSDP2 ranks, fixing the collective
    mismatch that occurs with vanilla ModuleList experts.
    """
    if not hasattr(model.config, "num_experts") and not hasattr(model.config, "num_local_experts"):
        return model  # Not a MoE model

    num_experts = getattr(model.config, "num_experts", None) or getattr(model.config, "num_local_experts", None)
    fused_count = 0

    for layer in model.model.layers:
        block = layer.mlp
        if not hasattr(block, "experts") or not isinstance(block.experts, nn.ModuleList):
            continue

        experts = block.experts
        gate_weights = torch.stack([experts[i].gate_proj.weight for i in range(num_experts)])
        up_weights = torch.stack([experts[i].up_proj.weight for i in range(num_experts)])
        down_weights = torch.stack([experts[i].down_proj.weight for i in range(num_experts)])

        fused = nn.Module()
        fused.gate_proj = nn.Parameter(gate_weights)
        fused.up_proj = nn.Parameter(up_weights)
        fused.down_proj = nn.Parameter(down_weights)
        block.experts = fused
        block.forward = _make_fused_forward(block)
        fused_count += 1

    if fused_count > 0:
        print(f"Fused experts in {fused_count} layers ({num_experts} experts each)")

    return model


def _make_fused_forward(block):
    """Create a forward method that uses fused expert weights."""

    def forward(hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = block.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, block.top_k, dim=-1)
        if block.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        expert_mask = F.one_hot(selected_experts, num_classes=block.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)

            eidx = expert_idx.item()
            gate_out = F.linear(current_state, block.experts.gate_proj[eidx])
            up_out = F.linear(current_state, block.experts.up_proj[eidx])
            current_hidden_states = F.linear(F.silu(gate_out) * up_out, block.experts.down_proj[eidx])
            current_hidden_states = current_hidden_states * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

    return forward


# ---- Main: same as trl/scripts/sft.py but with expert fusing ----


def main(script_args, training_args, model_args, dataset_args):
    from datasets import load_dataset
    from transformers import AutoConfig
    from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

    from trl import SFTTrainer, get_dataset, get_kbit_device_map, get_peft_config, get_quantization_config

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    valid_image_text_architectures = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values()

    if config.architectures and any(arch in valid_image_text_architectures for arch in config.architectures):
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    # Fuse MoE experts for FSDP2 compatibility
    model = fuse_moe_experts(model)

    if dataset_args.datasets and script_args.dataset_name:
        dataset = get_dataset(dataset_args)
    elif dataset_args.datasets and not script_args.dataset_name:
        dataset = get_dataset(dataset_args)
    elif not dataset_args.datasets and script_args.dataset_name:
        dataset = load_dataset(
            script_args.dataset_name, name=script_args.dataset_config, streaming=script_args.dataset_streaming
        )
    else:
        raise ValueError("Either `datasets` or `dataset_name` must be provided.")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()
    trainer.accelerator.print("Training completed.")
    trainer.save_model(training_args.output_dir)


def make_parser(subparsers=None, prog=None):
    from trl import DatasetMixtureConfig, ModelConfig, ScriptArguments, SFTConfig, TrlParser

    dataclass_types = (ScriptArguments, SFTConfig, ModelConfig, DatasetMixtureConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("sft", help="Run the SFT training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types, prog=prog)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args, dataset_args = parser.parse_args_and_config(fail_with_unknown_args=False)
    main(script_args, training_args, model_args, dataset_args)
