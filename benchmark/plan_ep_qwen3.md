# Plan: True Expert Parallelism for Qwen3 MoE in Transformers

## Goal

Add native EP support to Qwen3-30B-A3B in a local transformers fork so that
`DistributedConfig(enable_expert_parallel=True)` actually distributes experts across ranks.
Expected MFU improvement: 3-4% → 8-10% (matching the reference benchmark).

## Prerequisites

- TRL already has `--enable_expert_parallel` flag in SFTConfig (passes `DistributedConfig` to `from_pretrained`)
- TRL has `--fuse_moe_experts` that fuses weights into `[num_experts, ...]` tensors (proves the shape works)
- Transformers' `GroupedGemmParallel` and `RouterParallel` implement the EP sharding + routing

## Steps

### 1. Clone and setup

```bash
cd /fsx/amine_dirhoussi
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout v4.57.6  # match installed version
git checkout -b qwen3-moe-ep
```

### 2. Modify `configuration_qwen3_moe.py`

Add `base_model_ep_plan` class variable to `Qwen3MoeConfig`:

```python
base_model_ep_plan = {
    "layers.*.mlp.gate": "ep_router",
    "layers.*.mlp.experts.gate_proj": "grouped_gemm",
    "layers.*.mlp.experts.up_proj": "grouped_gemm",
    "layers.*.mlp.experts.down_proj": "grouped_gemm",
    "layers.*.mlp.experts": "gather",
    "layers.*.self_attn.q_proj": "colwise",
    "layers.*.self_attn.k_proj": "colwise",
    "layers.*.self_attn.v_proj": "colwise",
    "layers.*.self_attn.o_proj": "rowwise",
}
```

Reference: `src/transformers/models/gpt_oss/configuration_gpt_oss.py`

### 3. Modify `modeling_qwen3_moe.py`

Replace the MoE block with fused-expert architecture:

**3a. New `Qwen3MoeRouter` class** — wraps the gate linear, returns `(router_scores, router_indices)` in the format `RouterParallel` expects:
- `router_scores`: `(tokens, num_experts)` — full score matrix with zeros for non-selected
- `router_indices`: `(tokens, top_k)` — global expert indices

The `RouterParallel` EP hook will remap these to local expert indices per rank.

**3b. New `Qwen3MoeExperts` class** — holds fused `nn.Parameter` tensors:
- `gate_proj`: `[num_experts, moe_intermediate, hidden]`
- `up_proj`: `[num_experts, moe_intermediate, hidden]`
- `down_proj`: `[num_experts, hidden, moe_intermediate]`

Forward takes `(hidden_states, router_indices, routing_weights)` and loops over active experts using `F.linear(input, self.gate_proj[expert_idx])`.

`GroupedGemmParallel.partition_tensor` will automatically slice these along dim 0 by EP rank at load time.

**3c. Rewrite `Qwen3MoeSparseMoeBlock`** — uses the new Router + Experts:
- `self.gate = Qwen3MoeRouter(config)`
- `self.experts = Qwen3MoeExperts(config)`
- Forward: calls gate, then experts, returns `(hidden_states, router_logits)`

Reference: `src/transformers/models/gpt_oss/modeling_gpt_oss.py`

### 4. Weight conversion script

Create `scripts/convert_qwen3_moe_to_fused.py`:
- Loads sharded safetensors from HuggingFace Hub
- For each layer, stacks `experts.{i}.gate_proj.weight` → `experts.gate_proj` of shape `[128, 768, 2048]`
- Same for `up_proj` and `down_proj`
- `gate.weight` stays the same (shape `[num_experts, hidden]` is unchanged)
- Saves to `/fsx/amine_dirhoussi/Qwen3-30B-A3B-fused/`

Key: process shards sequentially to avoid OOM. Expert weights may be spread across shards.

### 5. Install and test

```bash
# Install modified transformers
cd /fsx/amine_dirhoussi/trl
source .venv/bin/activate
uv pip install -e /fsx/amine_dirhoussi/transformers

# Test forward (single GPU)
srun --nodes=1 --gpus-per-node=1 --partition=hopper-prod --time=0:30:00 bash -c '
source /fsx/amine_dirhoussi/trl/.venv/bin/activate
python -c "
import torch
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(\"/fsx/amine_dirhoussi/Qwen3-30B-A3B-fused\", dtype=torch.bfloat16)
model = model.cuda()
out = model(torch.tensor([[1,2,3,4,5]], device=\"cuda\"), use_cache=False)
print(f\"OK: {out.logits.shape}, nan={out.logits.isnan().any()}\")
"
'

# Test EP on 8 GPUs
srun --nodes=1 --gpus-per-node=8 --partition=hopper-prod --exclusive --mem=0 --time=0:30:00 bash -c '
source /fsx/amine_dirhoussi/trl/.venv/bin/activate
torchrun --nproc_per_node=8 test_ep.py
'

# Test SFT training with TRL
srun --nodes=1 --gpus-per-node=8 --partition=hopper-prod --exclusive --mem=0 --time=0:30:00 bash -c '
source /fsx/amine_dirhoussi/trl/.venv/bin/activate
accelerate launch --config_file /tmp/fsdp2.yaml trl/scripts/sft.py \
  --model_name_or_path /fsx/amine_dirhoussi/Qwen3-30B-A3B-fused \
  --enable_expert_parallel \
  --dataset_name THUDM/LongAlign-10k \
  --max_length 4096 --per_device_train_batch_size 1 \
  --gradient_checkpointing true --packing --packing_strategy wrapped \
  --max_steps 3 --logging_steps 1 --output_dir /tmp/ep_test --report_to none --save_strategy no
'
```

### 6. Submit benchmark

Update `benchmark/configs/qwen3_30b_a3b.yaml` to add EP runs using the fused model path.

## Key technical notes

- **RouterParallel** remaps `router_indices` from global to local expert IDs and slices `router_scores` to local experts only. The experts module forward must handle `num_local_experts` (= `num_experts / ep_size`) not `num_experts`.
- **GatherParallel** adds an all-reduce on the experts output to combine results across EP ranks.
- **Aux loss**: `output_router_logits=False` for benchmarking — the loss function expects global logits which EP hooks don't provide.
- **128 experts, EP=8**: each rank holds 16 experts. `128 % 8 == 0` ✓
- **FSDP2 + EP**: EP sharding happens at weight load time (`GroupedGemmParallel.partition_tensor`). FSDP2 wrapping happens later in accelerate. They compose — FSDP2 shards each rank's local expert weights across the DP group.
- **Checkpoint compat**: the gate weight key is identical (`gate.weight`). Expert weights change from `experts.{i}.gate_proj.weight` → `experts.gate_proj` (fused).

## Files to modify

| File | Action |
|---|---|
| `src/transformers/models/qwen3_moe/configuration_qwen3_moe.py` | Add `base_model_ep_plan` |
| `src/transformers/models/qwen3_moe/modeling_qwen3_moe.py` | Add `Qwen3MoeRouter`, `Qwen3MoeExperts`, rewrite `Qwen3MoeSparseMoeBlock` |
| `scripts/convert_qwen3_moe_to_fused.py` | New: weight conversion script |

## Files to reference (read-only)

| File | What to look at |
|---|---|
| `src/transformers/models/gpt_oss/configuration_gpt_oss.py` | `base_model_ep_plan` format |
| `src/transformers/models/gpt_oss/modeling_gpt_oss.py` | `GptOssExperts`, `GptOssTopKRouter`, `GptOssSparseMoeBlock` |
| `src/transformers/integrations/tensor_parallel.py:805` | `GroupedGemmParallel.partition_tensor` |
| `src/transformers/integrations/tensor_parallel.py:828` | `RouterParallel._prepare_output_fn` |
| `src/transformers/modeling_utils.py:2157` | `_ep_plan` property |
| `src/transformers/modeling_utils.py:5025` | `distribute_model` call |
