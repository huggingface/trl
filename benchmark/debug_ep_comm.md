# Expert Parallel Communication: allreduce vs all_to_all

## Current transformers EP implementation

The current EP in transformers uses **allreduce**, not all_to_all. This doc explains how it works, how it composes with DP/CP/FSDP2, and how it compares to "true EP" implementations (Megatron-LM, DeepSpeed-MoE, vLLM).

## Device mesh layout

EP reuses the `"tp"` mesh dimension. There is no separate EP dimension.

### 1 node, 8 GPUs, EP=8 (our profiled config)

```
device_mesh = init_device_mesh("cuda", (8,))

GPUs:  [0, 1, 2, 3, 4, 5, 6, 7]
        └─────── EP group ───────┘
```

All 8 GPUs form a single EP group. Each holds 128/8 = 16 local experts.

### 2 nodes, 16 GPUs, DP=2, EP=8

```
device_mesh = init_device_mesh("cuda", (2, 8), mesh_dim_names=("dp", "tp"))

        EP=8 (tp dim)
       ┌─────────────────────────────┐
GPU:   0   1   2   3   4   5   6   7     ← Node 0, EP group 0
GPU:   8   9  10  11  12  13  14  15     ← Node 1, EP group 1
       └dp┘└dp┘└dp┘└dp┘└dp┘└dp┘└dp┘└dp┘
```

- **EP group** = tp dimension (horizontal): 8 GPUs sharing experts, allreduce happens here
- **DP group** = dp dimension (vertical): same-role GPUs across replicas, synced by FSDP2

### With CP (e.g., 4 nodes, 32 GPUs, DP=2, CP=2, EP=8)

CP is handled by accelerate, not the device mesh passed to `from_pretrained`. The EP allreduce still operates over the tp dimension. CP splits the sequence across CP ranks, so the allreduce tensor is `(seq_len/cp_size, hidden_size)`.

## Where does the router live?

**Every rank has a full copy of the router.** The router is a small linear layer `(hidden_size, num_experts)` — for Qwen3-30B-A3B that's `(2048, 128)`. It is replicated, not sharded.

Since all ranks see identical hidden states (after FSDP2 allgather unshards them) and have the same router weights, every rank produces the **exact same routing decision**.

```
All 8 ranks see identical:
  router_indices: (seq, top_k=8)   — global expert IDs
  router_scores:  (seq, top_k=8)   — routing weights
```

## allreduce EP: step-by-step

Code: `RouterParallel._prepare_output_fn` and `MoeTensorParallelExperts` in `transformers/integrations/tensor_parallel.py`.

### Step 1: RouterParallel masks non-local experts

Each rank zeros out scores for experts it doesn't own, remaps global indices to local:

```
Router output (identical on all ranks):
  indices = [52, 42, 119, 67]     for some token, top_k=4
  scores  = [0.15, 0.12, 0.25, 0.08]

Rank 0 (owns experts 0-15):
  indices → [sentinel, sentinel, sentinel, sentinel]   ← none local
  scores  → [0, 0, 0, 0]

Rank 3 (owns experts 48-63):
  indices → [4, sentinel, sentinel, sentinel]          ← 52 % 16 = 4
  scores  → [0.15, 0, 0, 0]

Rank 7 (owns experts 112-127):
  indices → [sentinel, sentinel, 7, sentinel]          ← 119 % 16 = 7
  scores  → [0, 0, 0.25, 0]
```

### Step 2: Expert forward (local compute only)

Each rank runs only its local experts. Non-local tokens hit the sentinel index and produce zero output.

```
Rank 0: output = zeros                           (no local experts selected)
Rank 3: output = 0.15 * expert_4(hidden_states)  (partial)
Rank 7: output = 0.25 * expert_7(hidden_states)  (partial)
```

### Step 3: allreduce(SUM) via MoeTensorParallelExperts._prepare_output_fn

```python
return all_reduce_forward(outputs, device_mesh)   # line 1193
# calls dist.all_reduce(x, op=ReduceOp.SUM, group=device_mesh.get_group())
```

Sums partial outputs across all EP ranks:

```
final = rank0 + rank1 + ... + rank7
      = 0.15*expert_52(x) + 0.12*expert_42(x) + 0.25*expert_119(x) + 0.08*expert_67(x)
```

Every rank gets the full MoE output.

### Backward pass

Symmetric: `all_reduce_backward` on hidden_states and top_k_weights (identity in forward, allreduce in backward) ensures correct gradients flow to the router and upstream layers.

## Collectives per MoE layer (forward + backward)

| Direction | Collective | Tensor shape | Purpose |
|-----------|-----------|-------------|---------|
| Forward | `allreduce(SUM)` | `(seq, hidden_size)` | Sum partial expert outputs |
| Backward | `allreduce(SUM)` | `(seq, hidden_size)` | Gradient for hidden_states input |
| Backward | `allreduce(SUM)` | `(seq, top_k)` | Gradient for routing weights |

Total: **3 allreduce ops per MoE layer per step** (1 forward + 2 backward).

For Qwen3-30B-A3B with 48 layers (not all MoE — first layer is dense): roughly **~140 allreduce ops per training step**.

## How it composes with FSDP2

Full collective sequence for one MoE layer during forward:

```
FSDP2 allgather       ← unshard dense params (attention weights)
Attention forward     ← local compute
FSDP2 reduce_scatter  ← re-shard dense params

FSDP2 allgather       ← unshard MoE params (expert weights)
Router forward        ← local compute, replicated
RouterParallel mask   ← local, no communication
Expert forward        ← local compute on local experts
EP allreduce(SUM)     ← sum partial expert outputs across EP group
FSDP2 reduce_scatter  ← re-shard MoE params
```

In the Perfetto trace, you'll see this repeating pattern for each of the 48 layers. The FSDP2 collectives (allgather/reduce_scatter) are on the full world group, while the EP allreduce is on the tp sub-group.

## "True EP" with all_to_all (Megatron-LM / DeepSpeed-MoE)

### Step 1: Router (same — replicated)

### Step 2: all_to_all #1 — dispatch tokens to expert owners

```
Token T on rank 0 needs experts 52, 42, 119, 67:
  → Ship T to rank 3 (owns expert 52)
  → Ship T to rank 2 (owns expert 42)
  → Ship T to rank 7 (owns expert 119)
  → Ship T to rank 4 (owns expert 67)
```

Each rank RECEIVES tokens from other ranks that chose its local experts.

### Step 3: Local expert compute on received tokens only

### Step 4: all_to_all #2 — ship results back to originating rank

Each rank sends computed expert outputs back to the rank that owns the token. The originating rank does the weighted sum locally.

### Collectives per MoE layer

| Direction | Collective | Tensor shape | Purpose |
|-----------|-----------|-------------|---------|
| Forward | `all_to_all` | `(tokens_routed, hidden_size)` | Dispatch tokens to experts |
| Forward | `all_to_all` | `(tokens_routed, hidden_size)` | Return results to token owners |
| Backward | `all_to_all` × 2 | same | Reverse routing for gradients |

Total: **4 all_to_all ops per MoE layer per step**.

## Comparison

| | allreduce (transformers) | all_to_all (Megatron/DeepSpeed) |
|---|---|---|
| **Hidden states** | Replicated on ALL EP ranks | Each rank holds only its batch slice |
| **Router** | Replicated on all ranks | Replicated on all ranks |
| **Communication volume per MoE layer** | `allreduce(seq × hidden)` = full tensor | `2 × all_to_all(seq × top_k/num_experts × hidden)` = fraction |
| **Wasted compute** | Every rank runs expert forward on ALL tokens, zeros out non-local | Each rank only computes on tokens actually routed to it |
| **Memory** | Every rank holds all tokens' hidden states | Each rank holds only routed tokens |
| **Load balance sensitivity** | None — all ranks do same work regardless of routing | High — uneven routing = stragglers |
| **Implementation** | Simple — no token shuffling | Complex — variable token counts, padding, reordering |
| **Scaling** | Poor at large EP — allreduce over full hidden states grows with seq_len | Good — communication proportional to tokens_per_expert |

### Communication volume example (Qwen3-30B-A3B, seq=8192, hidden=2048, bf16)

**allreduce approach** (current):
```
Per MoE layer forward: allreduce of (8192, 2048) × 2 bytes = 32 MB
With 47 MoE layers: ~1.5 GB per forward pass
```

**all_to_all approach**:
```
Per MoE layer forward: 2 × all_to_all of (8192 × 8/128, 2048) × 2 bytes = 2 × 2 MB = 4 MB
With 47 MoE layers: ~188 MB per forward pass
```

The all_to_all approach moves **~8x less data** (ratio = num_experts / top_k / EP_size, here 128/8/1 ≈ depends on config). The gap widens with more experts and larger EP groups.

## What to look for in Perfetto traces

Open a trace from `benchmark/profiler_logs/ep_30b/run2_full/`.

1. **Find the allreduce pattern**: Search for `ncclAllReduce` in the NCCL stream. You should see clusters of allreduce calls — one per MoE layer.

2. **Measure compute vs communication**: Compare the duration of CUDA kernels (expert matmuls) vs NCCL ops. If allreduce dominates, EP overhead is the bottleneck.

3. **Compare ranks**: Open rank 0 and rank 4 side by side. In the allreduce approach, all ranks should have nearly identical timelines (same compute, same communication). If there's skew, something else (FSDP2, data loading) is the cause — not routing imbalance.

4. **FSDP2 overlap**: Check if FSDP2 allgather for the next layer overlaps with the current layer's compute. Good overlap means the FSDP2 prefetching is working.

5. **Memory (run2_full only)**: Look at the memory track to see if the allreduce buffers or expert activations dominate peak memory.
