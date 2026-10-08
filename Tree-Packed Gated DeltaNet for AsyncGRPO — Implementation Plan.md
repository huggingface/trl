# Tree-Packed Gated DeltaNet for AsyncGRPO

> Historical design plan. For the implemented code, benchmarks, and unresolved numerical limitations, start with
> [the review guide](benchmarks/README.md). Statements of intended exactness below are goals, not validation results.

## Goal

Extend TRL's existing tree packing so that it works efficiently and exactly with **Gated DeltaNet (GDN)** layers such as those used by Qwen3.5.

The current tree-packing implementation deduplicates repeated rollout prefixes for regular attention by forwarding each unique tree node once and using a FlexAttention ancestor mask:

```text
                         A1 -> A2
P1 -> P2 -> P3 --------+
                         B1 -> B2
```

For regular attention:

```text
A2 sees: P1 P2 P3 A1 A2
B2 sees: P1 P2 P3 B1 B2
```

and this is represented by:

```text
q sees k  <=>  k is an ancestor of q
```

with the current DFS interval test:

```text
tin[k] <= tin[q] < tout[k]
```

TRL's current PR forwards the shared tree once, preserves the original loss targets through `(token, next_token)` trie keys, and maps packed log-probabilities back to rollout tokens using `token_index`.

That approach works for full attention because visibility is controlled by an attention mask.

It does **not** directly work for Gated DeltaNet because GDN has no pairwise attention matrix to mask.

GDN instead carries a recurrent state:

\[
H_t = F(H_{t-1}, x_t)
\]

So at a tree branch:

```text
             P3
            /  \
           A1  B1
```

both branches must start from exactly the same state after `P3`:

```text
H_P3
 ├──> A1 -> A2
 └──> B1 -> B2
```

A flattened DFS sequence such as:

```text
P1 P2 P3 A1 A2 B1 B2
```

cannot simply be sent to the regular GDN kernel, because `B1` would incorrectly inherit state modified by `A1 A2`.

The goal is therefore:

> Compute every unique tree token once while making each node's GDN state depend on its tree parent rather than the previous packed token.

---

# 1. Important constraint: do not start with a token-wise tree kernel

The obvious mathematical implementation is:

```python
for node in topological_order:
    H[node] = gdn_update(
        H[parent[node]],
        x[node],
    )
```

This is useful as a **reference implementation**, but it should not be the production implementation.

Qwen3.5 does not normally train GDN using one sequential recurrent update per token.

Its long-sequence GDN path uses a **chunked gated delta-rule kernel** inherited from Qwen3-Next. The Transformers implementation exposes both:

```python
torch_recurrent_gated_delta_rule(...)
torch_chunk_gated_delta_rule(...)
```

and selects the chunked path for normal multi-token forward passes.

The optimized FLA implementation similarly exposes:

```python
chunk_gated_delta_rule(...)
fused_recurrent_gated_delta_rule(...)
```

but the fused recurrent implementation currently has no backward implementation, whereas the chunk path implements the complete training forward/backward.

So the production design should preserve the chunk formulation.

---

# 2. Current GDN execution model

The simplest recurrent form is approximately:

\[
H_t' = e^{g_t} H_{t-1}
\]

\[
\hat v_t = k_t^T H_t'
\]

\[
\Delta v_t = \beta_t (v_t - \hat v_t)
\]

\[
H_t = H_t' + k_t \Delta v_t^T
\]

\[
o_t = q_t^T H_t
\]

Transformers' reference implementation follows essentially this sequence.

Naively:

```text
H0
 |
token 0
 |
H1
 |
token 1
 |
H2
 |
token 2
 |
H3
...
```

This serial dependency is bad for training.

The chunk algorithm instead groups tokens, normally in chunks of 64:

```text
tokens

0 ........ 63 | 64 ...... 127 | 128 ...... 191
      C0              C1               C2
```

Within each chunk, the delta-rule recurrence is algebraically transformed into matrix operations.

The Transformers reference describes this as a UT transform that condenses multiple delta-rule updates into matmuls and triangular solves.

FLA's production path is broadly:

```python
g = gdn_gate_chunk_cumsum(...)

w, u, A = chunk_gated_delta_rule_fwd_intra(
    k=k,
    v=v,
    g=g,
    beta=beta,
    ...
)

h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
    k=k,
    w=w,
    u=u,
    g=g,
    initial_state=initial_state,
    ...
)

o = chunk_fwd_o(
    q=q,
    k=k,
    v=v_new,
    h=h,
    g=g,
    ...
)
```

The existing implementation therefore separates the work into roughly:

```text
             intra-chunk math
                    |
                    v
       KKT / UT / triangular solve
                    |
                    v
             W/U representation
                    |
                    v
          inter-chunk H recurrence
                    |
                    v
               output
```

The important seam for tree packing is:

```text
chunk_gated_delta_rule_fwd_h
```

because this is where the recurrent state is propagated between chunks.

---

# 3. What the existing state kernel does

For a normal sequence:

```text
             chunk 0      chunk 1      chunk 2
H_initial -------> H0 ----------> H1 ----------> H2
```

FLA launches state-processing programs over sequence/head/state tiles.

Inside one sequence, the kernel loops through that sequence's chunks while retaining pieces of the recurrent state in registers.

Conceptually:

```python
H = initial_state

for chunk in sequence:
    save_chunk_start_state(H)

    H = apply_chunk_transform(
        H,
        k_chunk,
        w_chunk,
        u_chunk,
        g_chunk,
    )

save_final_state(H)
```

This is crucial.

We do **not** want to replace it with:

```python
for each chunk in parallel:
    H_out[chunk] = F(H_out[parent_chunk], ...)
```

if that requires loading and storing the complete recurrent matrix from HBM for every chunk.

The current sequential-per-segment scan can keep `H` register-resident over many chunks.

That is one of the optimizations we should preserve.

---

# 4. Core design: represent the tree as linear segments

Do not expose every token as an independent recurrent node to the optimized kernel.

Instead, decompose the prefix tree into **maximal non-branching segments**.

Example:

```text
                         +-- A1 -> A2 -> A3
P1 -> P2 -> P3 -> P4 ---+
                         +-- B1 -> B2
```

becomes:

```text
segment 0:
P1 P2 P3 P4

segment 1:
A1 A2 A3

segment 2:
B1 B2
```

with:

```text
segment_parent = [
    -1,   # shared prefix
     0,   # A branch starts from segment 0
     0,   # B branch starts from segment 0
]
```

For deeper trees:

```text
P
|
S0
|
+------ A ------+
|               |
S1              B
|               |
+--- A1         S3
|
A2
|
S2
```

we produce:

```text
segment 0: P...
segment 1: A...
segment 2: A2...
segment 3: B...
```

plus:

```python
segment_parent = [-1, 0, 1, 0]
```

and ideally:

```python
segment_depth = [0, 1, 2, 1]
```

The recurrent semantics are now:

\[
H^{\text{in}}_s =
\begin{cases}
0 & \text{if root segment}\\
H^{\text{out}}_{\operatorname{parent}(s)}
& \text{otherwise}
\end{cases}
\]

and:

\[
H^{\text{out}}_s =
F_{\text{segment }s}(H^{\text{in}}_s)
\]

This is the key simplification.

Inside a segment, everything is still an ordinary linear GDN sequence.

---

# 5. Why segments are preferable to a chunk-parent tree

An alternative would be:

```text
C0 -> C1 -> C2
            / \
          A0   B0
          |    |
          A1   B1
```

with:

```python
chunk_parent = [-1, 0, 1, 2, 3, 2, 5]
```

That is mathematically fine.

But if every chunk independently loads:

```text
H[parent_chunk]
```

from global memory and stores:

```text
H[current_chunk]
```

we lose the current FLA optimization where multiple chunks share one live recurrent state.

Segments let us preserve:

```text
load H once
    |
    v
chunk
    |
chunk
    |
chunk
    |
chunk
    |
    v
store final H once
```

rather than:

```text
load H
chunk
store H

load H
chunk
store H

load H
chunk
store H
...
```

For a large GDN recurrent state, that difference matters a lot.

---

# 6. New packing metadata

TRL's tree packer already computes the tree topology.

Today it emits things like:

```text
packed input_ids
position_ids
tree_enter / tree_leave
shift_labels
token_index
segment_id
```

and uses DFS intervals for the FlexAttention ancestor relation.

Extend the packing output with GDN-specific topology.

A useful structure would be:

```python
@dataclass
class TreeGDNLayout:
    # Token order used by the packed representation.
    packed_token_ids: Tensor

    # Segment ranges into packed_token_ids.
    # segment i contains:
    # packed_token_ids[segment_start[i]:segment_end[i]]
    segment_start: Tensor[int32]
    segment_end: Tensor[int32]

    # Parent segment.
    # -1 means zero initial recurrent state.
    segment_parent: Tensor[int32]

    # Topological depth.
    segment_depth: Tensor[int32]

    # Segments ordered/grouped by depth.
    depth_offsets: Tensor[int32]
    depth_segment_ids: Tensor[int32]

    # Existing information.
    position_ids: Tensor
    shift_labels: Tensor
    token_index: Tensor

    # Existing tree information for full attention.
    tree_enter: Tensor
    tree_leave: Tensor
```

Example:

```text
Tree:

P1 P2 P3
      ├── A1 A2 A3
      │       └── C1 C2
      └── B1 B2


segments:

0: P1 P2 P3
1: A1 A2 A3
2: C1 C2
3: B1 B2


segment_parent:

[-1, 0, 1, 0]


segment_depth:

[0, 1, 2, 1]


depth_segment_ids:

[0, 1, 3, 2]


depth_offsets:

[0, 1, 3, 4]
```

Thus:

```text
depth 0 = segments [0]
depth 1 = segments [1, 3]
depth 2 = segments [2]
```

All segments at the same depth are independent and can be processed together.

---

# 7. Phase 1: write a token-level correctness reference

Before optimizing anything, build a minimal pure-PyTorch tree GDN.

This establishes the exact semantics.

For every tree node:

```python
def tree_gdn_reference(
    q,
    k,
    v,
    g,
    beta,
    parent,
):
    states = []
    outputs = []

    zero_state = ...

    for node in topological_order:
        p = parent[node]

        if p == -1:
            h = zero_state
        else:
            h = states[p]

        h = recurrent_step(
            h,
            q[node],
            k[node],
            v[node],
            g[node],
            beta[node],
        )

        states.append(h)
        outputs.append(read_state(h, q[node]))

    return torch.stack(outputs)
```

Do **not** optimize this.

Run it in fp32.

The reference test compares:

```text
A. original duplicated rollouts

P1 P2 P3 A1 A2
P1 P2 P3 B1 B2


B. tree recurrence

P1 P2 P3
        ├── A1 A2
        └── B1 B2
```

After gathering tree outputs back with `token_index`, verify:

```text
outputs
log-probs
loss
dQ
dK
dV
dg
dβ
model parameter gradients
```

against independent per-rollout forwards.

This test should become the semantic contract.

---

# 8. Phase 2: segment implementation using existing FLA kernels

This is the most important prototype.

Do **not write Triton yet**.

The existing FLA chunk GDN API already supports:

```python
initial_state=...
output_final_state=True
cu_seqlens=...
```

and `initial_state` has one recurrent state per input sequence.

This means every tree segment can already be represented as an independent varlen GDN sequence.

The only additional operation we need is:

```text
child initial_state = parent final_state
```

---

# 9. Process segments breadth-first

Suppose:

```text
              S0
            /    \
          S1      S2
         /  \
       S3   S4
```

Run:

```text
launch 1:
S0

launch 2:
S1 + S2

launch 3:
S3 + S4
```

Segments at one depth are flattened into one varlen batch.

For depth 1:

```text
S1 tokens = [A1 A2 A3]
S2 tokens = [B1 B2]
```

flatten as:

```text
[A1 A2 A3 B1 B2]
```

and:

```python
cu_seqlens = [0, 3, 5]
```

Initial state:

```python
initial_state = stack([
    final_state[S0],
    final_state[S0],
])
```

Then:

```python
outputs, final_state = chunk_gated_delta_rule(
    q=q_depth,
    k=k_depth,
    v=v_depth,
    g=g_depth,
    beta=beta_depth,

    initial_state=initial_state,
    output_final_state=True,

    cu_seqlens=cu_seqlens,
    ...
)
```

FLA already interprets varlen inputs as multiple sequences and expects the number of initial states to match the sequence count.

---

# 10. Prototype forward pseudocode

High-level version:

```python
def tree_chunk_gdn_reference(
    q,
    k,
    v,
    g,
    beta,
    layout: TreeGDNLayout,
):
    packed_output = torch.empty_like(v)

    # One final state per segment.
    segment_states = [None] * layout.num_segments

    for depth in range(layout.num_depths):
        segment_ids = layout.segments_at_depth(depth)

        q_level = []
        k_level = []
        v_level = []
        g_level = []
        beta_level = []

        cu_seqlens = [0]
        initial_states = []

        output_destinations = []

        for sid in segment_ids:
            start = layout.segment_start[sid]
            end = layout.segment_end[sid]

            q_level.append(q[:, start:end])
            k_level.append(k[:, start:end])
            v_level.append(v[:, start:end])
            g_level.append(g[:, start:end])
            beta_level.append(beta[:, start:end])

            cu_seqlens.append(
                cu_seqlens[-1] + end - start
            )

            parent = layout.segment_parent[sid]

            if parent == -1:
                initial_states.append(zero_state)
            else:
                initial_states.append(
                    segment_states[parent]
                )

            output_destinations.append((start, end))

        q_level = cat(q_level)
        k_level = cat(k_level)
        v_level = cat(v_level)
        g_level = cat(g_level)
        beta_level = cat(beta_level)

        initial_states = stack(initial_states)

        out_level, final_states = chunk_gated_delta_rule(
            q=q_level,
            k=k_level,
            v=v_level,
            g=g_level,
            beta=beta_level,
            initial_state=initial_states,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
        )

        scatter_outputs(
            packed_output,
            out_level,
            output_destinations,
        )

        for local_idx, sid in enumerate(segment_ids):
            segment_states[sid] = final_states[local_idx]

    return packed_output
```

This is intentionally simple.

The objective of Phase 2 is not to achieve optimal performance.

The objective is to prove:

```text
tree semantics
+
existing chunk GDN
+
autograd
```

work together exactly.

---

# 11. Important: avoid Python tensor-list overhead in the real prototype

The conceptual implementation above is fine for understanding.

The real prototype should generate flattened indices once in the packer.

Add:

```python
depth_token_indices
depth_token_offsets
depth_segment_offsets
depth_parent_segment
```

so the runtime does not repeatedly:

```python
cat(...)
stack(...)
build cu_seqlens(...)
```

inside each layer.

Remember that the same topology is used by **every GDN layer**.

The tree metadata must therefore be constructed once per packed batch and reused by all layers.

For example:

```python
@dataclass
class GDNExecutionPlan:
    # reordered token ids for processing all segments depth-by-depth
    gather_index: Tensor[int32]

    # inverse permutation to restore packed DFS order
    scatter_index: Tensor[int32]

    # all cu_seqlens concatenated
    cu_seqlens_per_depth: list[Tensor]

    # parent segment IDs for every depth
    parent_segments_per_depth: list[Tensor]

    # segment IDs returned by every depth
    output_segments_per_depth: list[Tensor]
```

Then each GDN layer only does gathers/scatters and kernel calls.

---

# 12. Consider changing the physical packed order

The existing attention implementation uses DFS order because it makes tree ancestry easy to represent.

GDN does not necessarily need that same physical order.

There are two choices.

## Choice A: preserve DFS layout

```text
P A C B ...
```

and gather each depth's segments before the GDN kernel.

Advantages:

```text
minimal change to existing TRL packing
full attention remains unchanged
loss indexing remains unchanged
```

Disadvantage:

```text
extra gathers/scatters for every GDN layer
```

## Choice B: create a GDN execution permutation

Keep the logical packed row unchanged, but construct:

```text
gdn_token_permutation
inverse_gdn_token_permutation
```

The GDN layer temporarily views tokens grouped by depth/segment:

```text
depth 0 segments
depth 1 segments
depth 2 segments
...
```

After GDN:

```text
scatter back to canonical packed order
```

Because Qwen3.5 alternates GDN and full-attention layers, hidden states eventually need to return to canonical tree order anyway.

Benchmark both.

The gather/scatter may be small relative to projection/GDN compute, but it should be measured.

---

# 13. Branch positions and GDN chunk boundaries

This is one of the most important implementation details.

Suppose FLA uses chunk size 64.

A segment might have:

```text
prefix segment length = 217
```

which means:

```text
64
64
64
25
```

The final partial chunk is already valid.

The segment ends exactly at the tree branch.

That is what we want.

Do not create a chunk that crosses from the shared prefix into one child:

```text
BAD

[P P P P P A A A ...]
          ^
          branch
```

because another child must fork from the state **before A**.

Instead, each segment independently terminates at the branch:

```text
GOOD

prefix segment:
[P P P P P]

branch A segment:
[A A A ...]

branch B segment:
[B B ...]
```

The existing chunk kernel handles the final partial chunk of each varlen sequence.

Therefore the simplest design is:

> Tree branches define sequence boundaries for GDN.

This does **not** mean every branch must align to a multiple of 64.

It only means a branch must terminate the current logical varlen sequence.

FLA can still internally mask/pad the tail chunk.

---

# 14. Why not force every branch to 64-token alignment?

Do not insert fake semantic padding into the tree just to make branches multiples of 64.

For example, if the shared prefix is 193 tokens:

```text
64 + 64 + 64 + 1
```

processing one 1-token tail chunk is not ideal, but is semantically simple.

The cost of occasionally underfilled final chunks is likely much lower than forwarding a long shared prefix several times.

Measure:

```text
average segment length
fraction of chunks that are partial
average valid tokens in final chunk
```

before adding any more complicated scheme.

---

# 15. The causal Conv1D must also become tree-aware

Qwen3.5's GDN path is not only the delta-rule recurrence.

The layer also uses short causal convolution before the recurrent GDN core, with `linear_conv_kernel_dim` defaulting to 4. Qwen3.5 inherits this GDN structure from Qwen3-Next.

For a regular sequence:

```text
x[t-3] x[t-2] x[t-1] x[t]
```

feed the convolution for token `t`.

At a branch:

```text
P1 P2 P3
      ├── A1
      └── B1
```

both `A1` and `B1` must use convolution history derived from:

```text
... P1 P2 P3
```

not:

```text
... P2 P3 A1
```

for B1.

Therefore each segment needs two kinds of initial state:

```text
1. GDN recurrent matrix H
2. short Conv1D history
```

Conceptually:

```python
SegmentState(
    recurrent_state=H,
    conv_state=conv_history,
)
```

At a branch:

```text
parent state
    |
    +------ copy/reference -----> child A
    |
    +------ copy/reference -----> child B
```

The convolution state is tiny compared with the GDN matrix, so duplicating it at branch boundaries is not a major concern.

---

# 16. Prototype Conv1D tree execution

Treat each segment as an independent sequence initialized with its parent's final convolution state.

Pseudo-interface:

```python
conv_out, final_conv_state = tree_segment_conv1d(
    x_segment,
    initial_conv_state=parent_conv_state,
    output_final_state=True,
)
```

If the current training convolution implementation cannot accept an initial state in chunk mode, write a small segment wrapper first.

Because the kernel width is only a few tokens, another acceptable correctness-first implementation is:

```text
prepend parent tail tokens to child segment
run ordinary causal convolution
discard outputs corresponding to prepended tokens
```

For kernel width `K`:

```text
child physical input:

[parent last K-1 tokens | child tokens]
```

then retain only:

```text
child outputs
```

For example, with kernel size 4:

```text
prefix:
... P7 P8 P9

child:
A1 A2 A3

temporary conv input:
P7 P8 P9 A1 A2 A3

returned outputs:
         A1 A2 A3
```

This duplicates at most three token activations per branch and is dramatically simpler than making the convolution kernel tree-aware immediately.

This is likely the correct first implementation.

---

# 17. Full Qwen3.5 layer execution

Qwen3.5 uses a hybrid stack of GDN and full-attention layers. The documented architecture uses three GDN layers for every full-attention layer.

So the model execution becomes:

```text
canonical packed tree hidden states
          |
          v
+-----------------------+
| GDN layer             |
|                       |
| tree-aware Conv1D     |
| tree segment GDN      |
+-----------------------+
          |
          v
canonical packed tree hidden states
          |
          v
+-----------------------+
| GDN layer             |
| tree segment GDN      |
+-----------------------+
          |
          v
+-----------------------+
| GDN layer             |
| tree segment GDN      |
+-----------------------+
          |
          v
+-----------------------+
| Full attention        |
|                       |
| existing tree FlexAttn|
+-----------------------+
          |
          v
...
```

Full-attention layers continue using the exact mechanism from PR #7328.

Only the linear-attention layers need the new tree state propagation.

---

# 18. Backward semantics

For a sequence:

```text
forward:

H0 -> H1 -> H2 -> H3


backward:

dH0 <- dH1 <- dH2 <- dH3
```

For a tree:

```text
             H_parent
             /      \
           HA        HB
           |         |
          LA        LB
```

the parent-state gradient is:

\[
\frac{\partial L}{\partial H_{\text{parent}}}
=
\frac{\partial L_A}{\partial H_{\text{parent}}}
+
\frac{\partial L_B}{\partial H_{\text{parent}}}
\]

So backward at a branch is simply:

```text
dH from child A -----+
                     |
                     +---- SUM ----> parent segment backward
                     |
dH from child B -----+
```

This is exactly the same fan-out/fan-in behavior normal autograd already implements.

---

# 19. Phase 2 backward: let PyTorch autograd do it

For the initial segment prototype, do not write any custom tree backward.

The computation graph is:

```python
parent_state = final_state_of_parent

child_A_out, child_A_state = chunk_gdn(
    ...,
    initial_state=parent_state,
)

child_B_out, child_B_state = chunk_gdn(
    ...,
    initial_state=parent_state,
)
```

`parent_state` feeds two differentiable operations.

Autograd naturally computes:

```text
grad(parent_state)
    =
grad_from_A
    +
grad_from_B
```

This is an important reason the segment prototype is so attractive.

We can validate the complete backward semantics before touching any Triton.

---

# 20. Make sure final states are not detached

Be careful with code like:

```python
segment_states[sid] = final_state.detach()
```

That would be wrong for training.

The graph must remain:

```text
prefix GDN
   |
final state
   |
   +-------- child A GDN
   |
   +-------- child B GDN
```

unless a custom autograd function explicitly reconstructs the corresponding gradient propagation.

The Phase 2 implementation should preserve the ordinary autograd graph.

---

# 21. Phase 3: optimize away repeated kernel orchestration

Once Phase 2 is correct and benchmarked, profile.

Expected overhead sources:

```text
one GDN launch group per tree depth
gathers/scatters
materializing final state at every segment boundary
duplicating parent states for siblings
small/short segments
Python/kernel launch overhead
```

Only optimize what appears in the profile.

Do not assume the state kernel is the bottleneck before measuring.

---

# 22. First custom-kernel target: segment-aware state propagation

The production FLA pipeline should remain:

```text
existing gate processing
        |
existing intra-chunk UT/KKT/WY
        |
NEW tree-segment state scan
        |
existing output kernel
```

The key functions today are:

```python
chunk_gated_delta_rule_fwd_intra(...)
chunk_gated_delta_rule_fwd_h(...)
chunk_fwd_o(...)
```

and backward similarly routes through:

```python
chunk_gated_delta_rule_bwd_dhu(...)
chunk_bwd_dqkwg(...)
prepare_wy_repr_bwd(...)
```


So ideally introduce:

```python
tree_chunk_gated_delta_rule_fwd_h(...)
```

rather than replacing the entire GDN implementation.

---

# 23. Proposed optimized API

Something approximately like:

```python
def tree_chunk_gated_delta_rule(
    q,
    k,
    v,
    g,
    beta,

    segment_cu_seqlens,
    segment_parent,
    segment_depth_offsets,

    scale=None,
    chunk_size=64,

    use_qk_l2norm_in_kernel=True,
):
    ...
```

or lower-level:

```python
tree_chunk_gated_delta_rule_fwd_h(
    k,
    w,
    u,
    g,

    segment_cu_seqlens,
    segment_parent,
    segment_depth_offsets,

    state_v_first=False,
    chunk_size=64,
)
```

The second interface is preferable if contributing upstream to FLA because it changes only the inter-chunk recurrence abstraction.

---

# 24. Forward kernel strategy

Do not try to process an arbitrary entire tree from one Triton program.

Instead, preserve independent linear scans.

At depth `d`, each segment gets a program that:

```text
load parent final state

for chunks in this segment:
    keep state in registers
    save per-chunk states required by backward/output
    update state

store segment final state
```

Pseudo-Triton:

```python
pid_segment = ...
pid_head = ...
pid_state_tile = ...

segment = level_segment_ids[pid_segment]

parent = segment_parent[segment]

if parent == -1:
    H = 0
else:
    H = load(segment_final_state[parent])

start_chunk = segment_chunk_start[segment]
end_chunk = segment_chunk_end[segment]

for chunk in range(start_chunk, end_chunk):
    store(
        chunk_initial_state[chunk],
        H,
    )

    H = apply_chunk_delta_update(
        H,
        k[chunk],
        w[chunk],
        u[chunk],
        g[chunk],
    )

store(
    segment_final_state[segment],
    H,
)
```

Launch this once per tree depth.

---

# 25. Why launch by depth?

If:

```text
S0
├── S1
│   └── S3
└── S2
```

then:

```text
S1 and S2
```

depend on:

```text
S0
```

and:

```text
S3
```

depends on `S1`.

A regular GPU kernel has no cheap global synchronization between arbitrary thread blocks.

Trying to process all depths in one ordinary Triton launch would require complex spin waiting / global synchronization and is not worth it initially.

Instead:

```text
kernel launch 0 -> depth 0
kernel launch 1 -> depth 1
kernel launch 2 -> depth 2
```

Kernel boundaries give the necessary synchronization.

GRPO trees should generally have far fewer branch depths than tokens.

---

# 26. Do not confuse tree depth with token depth

Tree depth here means **segment depth**, not token index.

A rollout may be:

```text
10,000-token shared prefix
    |
2,000-token branch
```

but only:

```text
segment depth = 2
```

So the number of required global synchronizations can remain tiny even for extremely long sequences.

This is why segment decomposition is useful.

---

# 27. State memory

The recurrent state is large relative to a normal token activation.

Its shape is approximately:

```text
[num_value_heads, key_head_dim, value_head_dim]
```

FLA supports this state explicitly as the `initial_state`/`final_state` for each sequence.

Therefore avoid storing a full recurrent state for every token.

The optimized implementation should store:

```text
per-chunk state
```

only where the existing backward requires it, plus:

```text
per-segment final state
```

for tree branching.

Do not introduce:

```text
per-token branch state
```

unless needed for a reference implementation.

---

# 28. Parent state duplication

Suppose one prefix fans out into eight generations:

```text
             prefix
  / / / / / / / /
 A B C D E F G H
```

Depth-1 execution needs eight logical copies of the same initial state.

Do not necessarily materialize:

```python
initial_states = parent_state.repeat(8, ...)
```

because the state is large.

The custom state kernel can instead read:

```python
parent = segment_parent[segment]
H = segment_final_state[parent]
```

Each child loads the same parent state independently.

For the PyTorch prototype, using:

```python
index_select(parent_states, parent_ids)
```

is fine.

For the optimized kernel, avoid a separate replicated state tensor.

---

# 29. Backward optimized design

The current FLA chunk path already implements a reverse recurrent kernel:

```python
chunk_gated_delta_rule_bwd_dhu(...)
```

that returns gradients associated with the recurrent-state path.

A tree version needs to change:

```text
one final dH flows backward through one sequence
```

into:

```text
sum dH from all children
then propagate through parent segment
```

Process segment depths in reverse:

```text
forward:

depth 0
depth 1
depth 2
depth 3


backward:

depth 3
depth 2
depth 1
depth 0
```

At every parent:

\[
dH^\text{out}_{parent}
=
\sum_{child \in children(parent)}
dH^\text{in}_{child}
+
dH^\text{local}_{parent}
\]

where `dH_local` includes gradients produced by outputs inside the parent segment.

---

# 30. Avoid atomics initially

One implementation is:

```python
atomic_add(
    dH_parent,
    dH_child,
)
```

for every child.

Avoid this as the first design.

The state is a matrix, potentially hundreds of thousands of values per segment.

Instead, because topology is known:

```text
parent -> child IDs
```

precompute:

```python
child_offsets
child_ids
```

similar to CSR:

```python
child_offsets[parent]:
    range into child_ids
```

Then run a branch-gradient reduction:

```python
for parent in parallel:
    dH_from_children = 0

    for child in children[parent]:
        dH_from_children += dH_child[child]

    dH_parent += dH_from_children
```

Then run the ordinary reverse segment scan.

For GRPO where fanout is typically `num_generations`, this reduction structure should be straightforward.

---

# 31. Potential backward execution

For depth `d`:

```text
1. gather/sum incoming dH from children
2. add any external final-state gradient
3. run reverse GDN state scan for every segment at depth d
4. produce dH at segment input
5. pass that dH to its parent
```

Pseudo:

```python
for depth in reversed(range(num_depths)):
    segments = segments_at_depth(depth)

    dht = reduce_child_state_grads(
        segments,
        child_ids,
        child_offsets,
        child_dh0,
    )

    dh_chunks, dh0, dv = tree_segment_bwd_dhu(
        ...,
        dht=dht,
    )

    segment_input_grad[segments] = dh0
```

Then the rest of the local GDN backward can reuse FLA's existing kernels.

---

# 32. Gradient checkpointing

TRL training may run with model gradient checkpointing.

The execution plan must therefore be deterministic and reconstructible.

Do not create dynamic GPU-side tree scheduling based on runtime values.

All topology should be derived from the packed batch:

```text
segment_start
segment_end
segment_parent
segment_depth
cu_seqlens
permutations
```

and remain available during recomputation.

This makes gradient checkpointing straightforward:

```text
recomputed forward sees exactly the same segment graph
```

---

# 33. Interaction with `(token, next_token)` trie semantics

Do not change this part of PR #7328.

The current tree uses `(token, next_token)` rather than only `token` to ensure every packed position has exactly one training target.

Example:

```text
rollout 1:
P1 P2 P3 A B

rollout 2:
P1 P2 P3 C D
```

The tree duplicates the last shared predictor token so one copy predicts `A` and another predicts `C`.

This costs one extra forwarded token per extra branch but preserves exact LM-head semantics.

GDN should consume the same logical prefix forest.

If the tree duplicates `P3` for label reasons, those copies should also get identical parent recurrent state and therefore produce identical hidden states before their different next-token labels are applied.

That remains correct.

---

# 34. Position IDs

Each duplicated tree node retains the original rollout position ID.

For GDN itself position IDs do not control recurrence in the way RoPE controls full attention, but other layers still rely on consistent token positions.

Do not derive position from:

```text
packed DFS index
```

or:

```text
segment-local index
```

Use the existing original-rollout position IDs.

---

# 35. Loss path should remain unchanged

The entire point is to make GDN another implementation of the same packed logical tree.

The loss path from PR #7328 should remain:

```text
packed hidden states
        |
fused LM head using shift_labels
        |
packed log_probs
        |
log_probs[token_index]
        |
original rollout order
        |
GRPO loss
```

Do not introduce a second GDN-specific loss mapping.

The model should return hidden states in the existing canonical packed-token order.

---

# 36. Integration point in Transformers

Qwen3.5's GDN class inherits from the Qwen3-Next implementation.

The current forward computes projections, convolution, `beta`, decay `g`, then chooses between the recurrent and chunk gated-delta-rule functions.

For normal multi-token execution it calls the chunked function.

The clean integration is therefore to add an optional tree execution descriptor:

```python
tree_gdn_layout=None
```

to the GDN path.

Conceptually:

```python
if tree_gdn_layout is None:
    core_attn_out = torch_chunk_gated_delta_rule(...)
else:
    core_attn_out = tree_chunk_gated_delta_rule(
        ...,
        layout=tree_gdn_layout,
    )
```

Do not overload `attention_mask` with this information.

GDN does not use an attention mask.

Make the state topology explicit.

---

# 37. Avoid making tree packing Qwen-specific

The semantics are generic:

```text
recurrent linear attention
+
tree-shaped prefix sharing
```

So the model-facing interface should preferably represent:

```text
recurrent_state_graph
```

rather than:

```text
qwen35_tree_data
```

A generic structure might be:

```python
@dataclass
class RecurrentTreeLayout:
    segment_start: Tensor
    segment_end: Tensor
    segment_parent: Tensor
    segment_depth: Tensor
    depth_offsets: Tensor
```

Any recurrent model with:

```text
state_out = F(sequence, state_in)
```

could potentially use it.

---

# 38. Integration with TRL tree packing

Current flow:

```text
rollouts
   |
planner
   |
TreePacking.pack
   |
prefix forest
   |
packed IDs
position IDs
DFS stamps
shift labels
token index
   |
model
   |
tree FlexAttention
   |
loss
```

New flow:

```text
rollouts
   |
planner
   |
TreePacking.pack
   |
prefix forest
   |
   +--> canonical packed tokens
   |
   +--> full-attention metadata
   |      tree_enter
   |      tree_leave
   |
   +--> recurrent metadata
   |      segment_start
   |      segment_end
   |      segment_parent
   |      segment_depth
   |
   +--> loss metadata
          shift_labels
          token_index
   |
model
   |
   +--> full-attention layer
   |      FlexAttention ancestor mask
   |
   +--> GDN layer
          segment-state execution
   |
loss
```

---

# 39. Planner cost model

The current planner already reasons about the number of **unique forwarded tree tokens** rather than the sum of duplicated rollout lengths.

For GDN, token sharing still saves projection/MLP/GDN work.

However, tree GDN introduces additional costs not present in full attention:

```text
branch recurrent-state materialization
extra partial chunks
depth launch synchronization
conv-history duplication
```

Eventually the planner might estimate:

\[
C_\text{GDN-tree}
\approx
aN_\text{unique tokens}
+
bN_\text{segments}
+
cN_\text{partial chunks}
\]

but do **not** change the planner initially.

First measure whether unique-token count remains a sufficiently good predictor.

---

# 40. Tests: tiny semantic trees

Add extremely small hand-written trees.

## No branch

```text
A B C D
```

Tree output must match regular sequence GDN.

## Single branch

```text
A B
   ├── C D
   └── E F
```

Compare against:

```text
A B C D
A B E F
```

## Nested branch

```text
A B
   ├── C
   │   ├── D E
   │   └── F
   └── G
```

## Branch at token 1

Tests initial-state handling.

## Branch exactly at chunk boundary

For chunk size 64:

```text
prefix length = 64
```

## Branch one token before boundary

```text
prefix length = 63
```

## Branch one token after boundary

```text
prefix length = 65
```

These three are particularly important.

---

# 41. Tests: outputs

For every logical rollout compare packed-tree output against independent sequence execution.

Use:

```python
torch.testing.assert_close(...)
```

for:

```text
GDN core output
decoder-layer output
model hidden states
LM logits/log-probs
```

Start with fp32 reference tolerance.

Then test bf16 separately with realistic tolerances.

---

# 42. Tests: gradients

This is mandatory.

Compare gradients for:

```text
input hidden states
Q projection weights
K projection weights
V projection weights
gate parameters
beta projection
output projection
MLP parameters
embedding / LM head if applicable
```

The test should be:

```text
duplicated rollout execution
vs
tree execution
```

with the **same logical loss terms**.

The current tree-attention PR already treats gradient equivalence as a critical correctness property; do the same for GDN.

---

# 43. Tests: state fan-out gradient

Create a dedicated test where one parent state feeds many children:

```text
        P
   / / / \ \ \
  A B C D E F
```

Compute:

```python
loss = (
    loss_A
    + loss_B
    + loss_C
    + loss_D
    + loss_E
    + loss_F
)
```

Verify:

```text
gradient of prefix parameters
```

equals the sum of gradients from individually executed rollouts.

This specifically validates branch-state gradient accumulation.

---

# 44. Tests: Conv1D branch history

Use a tiny deterministic convolution where the effect of previous tokens is obvious.

For example:

```text
prefix: 1 2 3
branch A: 10
branch B: 20
```

Verify both branches see history:

```text
1 2 3
```

and B does not see A.

This should be tested separately from GDN.

---

# 45. Tests: hybrid Qwen3.5

Once standalone GDN passes:

```text
tiny Qwen3.5 config
```

with the actual layer pattern:

```text
GDN
GDN
GDN
full attention
```

Compare:

```text
independent rollouts
vs
tree-packed rollouts
```

for:

```text
final hidden states
log-probs
loss
all parameter gradients
```

This catches interactions between:

```text
tree GDN
+
tree FlexAttention
+
residuals
+
MLP
```

---

# 46. Benchmark matrix

Use several sharing regimes.

```text
Case A:
short prompt
long completions
low sharing

Case B:
long prompt
short completions
high sharing

Case C:
long prompt
long completions
medium sharing

Case D:
multi-turn agent rollout
many repeated conversation prefixes

Case E:
nested branch tree
```

Vary:

```text
num_generations:
2 / 4 / 8 / 16

prompt length:
1k / 4k / 16k / 64k

completion length:
256 / 1k / 4k

branch depth:
1 / 2 / 4 / 8
```

---

# 47. Metrics

Measure at least:

```text
unique forwarded tokens
logical trained tokens
packing ratio

total fwd+bwd time

GDN time
full-attention time
MLP time

GDN projection time
Conv1D time
GDN intra-chunk time
GDN H-scan time
tree state-management time

number of segments
mean segment length
median segment length

number of partial chunks
average final-chunk occupancy

tree depth
kernel launches per GDN layer

peak allocated memory
```

And the most important throughput metrics:

\[
\text{logical trained tokens / second}
\]

and:

\[
\text{rollouts / second}
\]

not only raw packed-forward time.

---

# 48. Performance model

Let:

```text
D = total duplicated rollout tokens
U = unique tree tokens
S = number of segments
R = recurrent-state size
L = segment-tree depth
```

Normal execution is approximately:

\[
T_\text{normal}
\approx
C_\text{token} D
\]

Tree GDN is approximately:

\[
T_\text{tree}
\approx
C_\text{token} U
+
C_\text{state} S R
+
C_\text{launch} L
+
C_\text{partial}
\]

Tree packing wins when:

\[
C_\text{token}(D-U)
>
C_\text{state}SR
+
C_\text{launch}L
+
C_\text{partial}
\]

This is the benchmark question.

For long repeated GRPO prefixes, `D-U` can be extremely large.

---

# 49. Do not optimize into one persistent DFS kernel too early

A tempting design is:

```text
one persistent program

process prefix
push H
process child A
restore H
process child B
restore H
...
```

This resembles the existing DFS tree order.

Avoid it initially.

Problems:

```text
large recurrent state
branch states cannot all stay in registers
spills become unavoidable
branch lengths diverge
poor GPU load balance
backward becomes substantially harder
existing FLA chunk kernels cannot be reused cleanly
```

The segment approach deliberately trades a few kernel launches for much more reuse of mature kernels.

---

# 50. Future possibility: tree scan over affine chunk transforms

There may eventually be a more elegant formulation.

A GDN chunk can be viewed abstractly as a transform:

\[
H_\text{out}
=
\mathcal{T}_c(H_\text{in})
\]

and the chunked delta-rule algebra produces a compact representation of this transform.

For a sequence:

\[
H_3
=
\mathcal T_3
\circ
\mathcal T_2
\circ
\mathcal T_1
(H_0)
\]

For a tree:

```text
                  TA
                 /
TPrefix --------
                 \
                  TB
```

we need:

\[
H_A =
\mathcal T_A
\circ
\mathcal T_P
(H_0)
\]

and:

\[
H_B =
\mathcal T_B
\circ
\mathcal T_P
(H_0)
\]

This starts looking like a parallel tree-prefix scan over composable state transforms.

That may eventually produce an even better kernel.

Do not start there.

First establish that the simpler segment representation wins in the actual GRPO workload.

---

# 51. Implementation milestones

## Milestone 0 — topology only

Extend `TreePacking` with:

```text
segment_start
segment_end
segment_parent
segment_depth
depth_offsets
```

Add visualization/debug helper:

```text
segment 0 [0:1234] parent=-1 depth=0
segment 1 [1234:1700] parent=0 depth=1
segment 2 [1700:2144] parent=0 depth=1
...
```

Verify segment paths reconstruct every original rollout.

---

## Milestone 1 — pure recurrent reference

Implement:

```python
tree_recurrent_gdn_reference(...)
```

one token at a time.

Only correctness matters.

Compare against duplicated independent sequences.

---

## Milestone 2 — standalone FLA segment prototype

Implement:

```python
tree_chunk_gated_delta_rule(...)
```

as multiple existing:

```python
chunk_gated_delta_rule(...)
```

calls grouped by tree depth.

Do not modify FLA.

Use autograd.

Benchmark standalone GDN.

---

## Milestone 3 — Conv1D tree handling

Add segment-aware short-convolution state.

Simplest initial strategy:

```text
prepend parent K-1 history tokens
run normal conv
discard prefix outputs
```

Validate independently.

---

## Milestone 4 — Qwen3.5 model integration

Route:

```text
linear_attention layer -> tree GDN
full_attention layer   -> existing tree FlexAttention
```

Keep the remainder of the model unchanged.

Run tiny-model gradient parity.

---

## Milestone 5 — real AsyncGRPO integration

Run:

```text
Qwen3.5
AsyncGRPO
packing="sequence"

vs

Qwen3.5
AsyncGRPO
packing="tree"
```

using identical stored rollouts first.

Measure:

```text
logical tokens/s
fwd_bwd_s
memory
packing ratio
```

Do this before live asynchronous generation so rollout differences do not contaminate benchmarking.

---

## Milestone 6 — profile

Identify whether overhead comes primarily from:

```text
tree-depth kernel launches
state copying
token gathers/scatters
partial chunks
Conv1D
GDN state scan
```

Only then decide what to fuse.

---

## Milestone 7 — custom FLA state kernel

Implement:

```python
tree_chunk_gated_delta_rule_fwd_h
```

that loads parent state directly using `segment_parent`.

Keep existing:

```text
intra-chunk GDN
output computation
```

unchanged.

---

## Milestone 8 — custom backward

Implement reverse segment-depth execution.

At branch joins:

```text
sum child dH
```

before running the parent reverse scan.

Avoid global atomics unless profiling proves them worthwhile.

---

# 52. Recommended first code structure

TRL:

```text
trl/experimental/async_grpo/tree/
    tree.py
    packing.py
    flex_attention.py

    recurrent_layout.py       # new
```

Potential API:

```python
class RecurrentTreeLayout:
    segment_start
    segment_end
    segment_parent
    segment_depth
    depth_offsets
    depth_segment_ids
```

Model integration prototype:

```text
trl/experimental/async_grpo/tree_gdn.py
```

with:

```python
tree_chunk_gated_delta_rule
tree_conv1d
```

Initially keep this outside Transformers/FLA while iterating.

Once the design is validated, move the generic primitive into the lower-level library.

---

# 53. Debug representation

Make topology easy to inspect.

For every packed row optionally print:

```text
Tree:

S0 parent=-1 depth=0
|  tokens: P1 P2 P3 P4
|
+-- S1 parent=0 depth=1
|   tokens: A1 A2
|   |
|   +-- S3 parent=1 depth=2
|       tokens: C1 C2
|
+-- S2 parent=0 depth=1
    tokens: B1 B2 B3
```

Also print the equivalent reconstructed rollout paths:

```text
rollout 0:
S0 -> S1 -> S3

rollout 1:
S0 -> S2
```

This will make correctness bugs vastly easier to diagnose than staring at flattened indices.

---

# 54. Assertions worth keeping in debug builds

Assert:

```python
segment_parent[s] < s
```

if segments are stored topologically.

Assert:

```python
segment_depth[s] == (
    0
    if parent == -1
    else segment_depth[parent] + 1
)
```

Assert every packed token belongs to exactly one segment.

Assert every original rollout can be reconstructed from its leaf by walking parents.

Assert original token/position sequence matches the reconstructed segment path.

Assert no child segment begins before its parent ends semantically.

These are cheap and catch topology bugs before they become silent gradient bugs.

---

# 55. The smallest viable experiment

Before touching the whole trainer, implement this isolated benchmark:

```text
shared prefix:
4096 tokens

8 branches:
1024 tokens each
```

Baseline:

```text
8 independent sequences
length = 5120
```

Tree:

```text
segment 0:
4096 shared tokens

depth 1:
8 × 1024-token segments
```

Logical tokens:

```text
8 * 5120 = 40,960
```

Unique forwarded tokens:

```text
4096 + 8 * 1024
= 12,288
```

Ideal token-compute reduction:

```text
3.33x
```

Run only one standalone GDN layer.

Compare:

```text
forward
forward + backward
peak memory
gradient parity
```

If the segment prototype cannot win substantially in this regime, inspect why before integrating it into Qwen3.5.

---

# 56. Then test the opposite regime

Shared prefix:

```text
128
```

Branches:

```text
8 × 4096
```

Baseline tokens:

```text
33,792
```

Tree tokens:

```text
32,896
```

Almost no sharing.

The tree path should probably be slower.

That is fine.

Eventually the trainer can decide whether to use tree GDN based on expected packing ratio.

The optimization does not need to dominate for workloads with no prefix sharing.

---

# 57. Success criteria

Correctness:

```text
tree outputs ~= independent rollout outputs

tree loss ~= independent rollout loss

tree gradients ~= independent rollout gradients
```

Performance:

For high-prefix-sharing agentic workloads:

```text
logical trained tokens/s
```

must improve materially.

The goal is **not**:

```text
make one packed step faster than one baseline step
```

because the packed step may contain several times more training data.

The same reasoning already applies to PR #7328.

---

# 58. Final recommended design

The architecture I would pursue is:

```text
                    TRL TREE PACKER
                           |
            +--------------+--------------+
            |                             |
            v                             v
    FULL ATTENTION METADATA        RECURRENT METADATA
      tin / tout                segment ranges / parents
            |                             |
            v                             v
     FlexAttention                 GDN segment executor
    ancestor masking                       |
                                    depth 0 segments
                                           |
                                    depth 1 segments
                                           |
                                    depth 2 segments
                                           |
                                           v
                              existing FLA chunk machinery
```

Inside GDN:

```text
Q/K/V/g/beta
     |
     v
existing chunk-local UT/KKT/WY preprocessing
     |
     v
NEW tree-segment recurrent H propagation
     |
     v
existing chunk output computation
```

Backward:

```text
existing local backward
        |
        v
child segment dH
        |
        v
SUM at tree branch
        |
        v
parent reverse segment scan
        |
        v
existing local parameter/input gradients
```

The implementation order should be:

```text
1. token-wise reference
2. segment topology
3. existing-FLA segment prototype
4. Conv1D branch handling
5. Qwen3.5 integration
6. real AsyncGRPO benchmark
7. profile
8. custom tree state kernel
9. custom reverse tree state kernel
```

The most important principle is:

> **Do not rewrite Gated DeltaNet. Reuse its optimized chunk-local algebra and change only how recurrent state flows between independent linear segments of the rollout tree.**

That gives the smallest correctness surface, lets existing autograd validate the idea before Triton work, and targets exactly the part of the kernel where sequence semantics currently assume `previous chunk` instead of `parent segment`.
