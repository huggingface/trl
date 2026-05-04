# Study notes: how a NaN in `_grouped_mm`'s backward poisons a whole MoE step

A self-contained, runnable walkthrough of the autograd chain that produces the bug. Each section is structured **math → torch → tiny example you can paste into a REPL**.

The whole point: understand why the wrapper-side `masked_fill` fix is correct, by tracing one NaN through the chain rule.

---

## 0. Setting

Inside the MoE wrapper, the relevant slice of code is:

```python
# Forward
selected = hidden_states[idx]                 # (S, D)  fancy gather
proj_out = grouped_mm(selected, W, offsets)   # (S, D') — kernel only writes [0, offsets[-1])
proj_out = SwiGLU(proj_out)
proj_out = grouped_mm(proj_out, W2, offsets)  # (S, D)
weighted = proj_out * sample_weights          # (S, D)
final = weighted.view(T, K, D).sum(dim=1)     # (T, D)
loss = ...
```

with shapes:
- `hidden_states` (= `H`): `(T, D)`, `T` = num tokens
- `idx`: `(S,)` with values in `[0, T)`. Here `S = T × K` and `K = top_k`.
- `selected` (= `Y`): `(S, D)`
- `offsets[-1]` < `S` because some slots are EP "sentinels" the kernel skips
- `sample_weights[sentinel]` is already 0 (`RouterParallel` masked it upstream)

The bug we'll trace: for one row `r ∈ [offsets[-1], S)` (a sentinel), `_grouped_mm`'s backward leaves `dY[r, :]` uninitialized (NaN). We follow that NaN through the chain rule until it lands as NaN inside `dH`, and we see how `masked_fill` placed correctly absorbs it.

---

## 1. The forward operation `Y = H[idx]` — fancy indexing

### Math

For each output row `r ∈ [0, S)` and feature `j ∈ [0, D)`:

$$Y[r, j] \;=\; H[\text{idx}[r], j]$$

This is just **copying rows of `H`, possibly with repetition**. Each `Y[r, :]` is a duplicate of `H[idx[r], :]`. If two rows of `Y` share the same `idx`, they're literally the same data — so the operation is many-to-one (one source can feed many sinks).

### Torch

```python
Y = H[idx]
```

This is "fancy indexing" or "advanced indexing." Even though `H[idx]` looks like a slice, it's a **copy** of the picked rows, not a view — because `idx` may produce a non-contiguous, non-strided pattern.

Equivalent explicit form:

```python
Y = torch.empty(idx.shape + H.shape[1:], dtype=H.dtype, device=H.device)
for r in range(len(idx)):
    Y[r] = H[idx[r]]
```

### Tiny example

```python
import torch

H = torch.tensor([[1., 2.],
                  [3., 4.],
                  [5., 6.]], requires_grad=True)
idx = torch.tensor([0, 0, 1, 1, 2, 2])
Y = H[idx]
print(Y)
# tensor([[1., 2.],
#         [1., 2.],
#         [3., 4.],
#         [3., 4.],
#         [5., 6.],
#         [5., 6.]], grad_fn=<IndexBackward0>)
```

`Y[0]` and `Y[1]` are both copies of `H[0]`. They aren't the *same memory* — they're two separate copies. But autograd remembers that both came from `H[0]`, which is what step 2 will exploit.

---

## 2. The backward of fancy indexing — scatter-add

### Math (chain rule)

We want `∂L/∂H[t, j]` for each token `t` and feature `j`. Apply the chain rule, summing over the contributions of every output element:

$$\frac{\partial L}{\partial H[t, j]} \;=\; \sum_{r=0}^{S-1}\, \sum_{k=0}^{D-1}\, \underbrace{\frac{\partial L}{\partial Y[r, k]}}_{\text{= dY}[r,k]} \cdot \underbrace{\frac{\partial Y[r, k]}{\partial H[t, j]}}_{\text{depends on } \text{idx}[r], k, t, j}$$

Since `Y[r, k] = H[idx[r], k]` is just a copy:

$$\frac{\partial Y[r, k]}{\partial H[t, j]} \;=\; \mathbb{1}[\text{idx}[r] = t]\;\cdot\;\mathbb{1}[k = j]$$

Both indicators must be 1 for any contribution. Plugging in:

$$\frac{\partial L}{\partial H[t, j]} \;=\; \sum_{r \,:\, \text{idx}[r] = t}\, \frac{\partial L}{\partial Y[r, j]}$$

In tensor form:

$$dH[t, :] \;=\; \sum_{r \,:\, \text{idx}[r] = t}\, dY[r, :]$$

**Token `t`'s gradient is the SUM of every `dY` row whose source was `t`.**

This makes physical sense: if `H[t]` was duplicated into multiple `Y` rows, each duplicate contributes back to `H[t]` via its own gradient.

### Torch

PyTorch implements this with `Tensor.index_add_`:

```python
def gather_backward(dY, idx, H_shape):
    dH = torch.zeros(H_shape, dtype=dY.dtype, device=dY.device)
    dH.index_add_(0, idx, dY)
    return dH
```

`dH.index_add_(0, idx, dY)` is the in-place version of:

```python
for r in range(len(idx)):
    dH[idx[r]] += dY[r]
```

**The `+=` is doing the sum from the math.** When two slots `r1, r2` share the same source `idx[r1] = idx[r2] = t`, both gradients land on `dH[t]` and add up.

### Tiny example

Continuing from §1's setup:

```python
# Pretend the loss gives us this dY (gradient w.r.t. Y):
dY = torch.tensor([[0.1, 0.2],   # slot 0 came from H[0]
                   [0.3, 0.4],   # slot 1 came from H[0]
                   [0.5, 0.6],   # slot 2 came from H[1]
                   [0.7, 0.8],   # slot 3 came from H[1]
                   [0.9, 1.0],   # slot 4 came from H[2]
                   [1.1, 1.2]])  # slot 5 came from H[2]

# Drive backward by hand
Y.backward(dY)
print(H.grad)
# tensor([[0.4, 0.6],   # 0.1+0.3, 0.2+0.4   ← both slots from token 0 sum
#         [1.2, 1.4],   # 0.5+0.7, 0.6+0.8
#         [2.0, 2.2]])  # 0.9+1.1, 1.0+1.2
```

Verify by hand:
- `H.grad[0] = dY[0] + dY[1] = [0.1, 0.2] + [0.3, 0.4] = [0.4, 0.6]` ✓
- `H.grad[1] = dY[2] + dY[3] = [0.5, 0.6] + [0.7, 0.8] = [1.2, 1.4]` ✓
- `H.grad[2] = dY[4] + dY[5] = [0.9, 1.0] + [1.1, 1.2] = [2.0, 2.2]` ✓

That `+=` is the whole story.

---

## 3. NaN propagation through `+=`

### Math

IEEE 754 floating-point semantics:

- `real + NaN = NaN`
- `NaN + real = NaN`
- `NaN + NaN = NaN`

There is no real number `x` for which `x + NaN ≠ NaN`. **A single NaN, once added in, propagates to every subsequent partial sum at that position.**

### Torch

`Tensor.__iadd__` and `index_add_` follow the IEEE rule. There's no special NaN handling.

### Tiny example: one NaN destroys a whole token's gradient

```python
import torch

H = torch.tensor([[1., 2.],
                  [3., 4.],
                  [5., 6.]], requires_grad=True)
idx = torch.tensor([0, 0, 1, 1, 2, 2])
Y = H[idx]

# Pretend slot 1 is a "sentinel" whose gradient was left uninitialized → NaN
dY = torch.tensor([[0.1,    0.2],
                   [float('nan'), float('nan')],   # ← sentinel row
                   [0.5,    0.6],
                   [0.7,    0.8],
                   [0.9,    1.0],
                   [1.1,    1.2]])

Y.backward(dY)
print(H.grad)
# tensor([[nan, nan],   ← token 0: real(0.1) + NaN = NaN
#         [1.2, 1.4],
#         [2.0, 2.2]])
```

Even though only ONE row of `dY` is NaN, **token 0's entire gradient is now NaN** — the `+=` polluted it.

Now imagine sentinel slots `1, 3, 5`:

```python
dY = torch.tensor([[0.1,    0.2],
                   [float('nan'), float('nan')],
                   [0.5,    0.6],
                   [float('nan'), float('nan')],
                   [0.9,    1.0],
                   [float('nan'), float('nan')]])

H.grad = None  # reset
Y = H[idx]
Y.backward(dY)
print(H.grad)
# tensor([[nan, nan],     ← token 0
#         [nan, nan],     ← token 1
#         [nan, nan]])    ← token 2
```

Every token has at least one sentinel slot in this layout, so every row of `H.grad` is NaN. From here, `H.grad` flows back through every previous transformer block (residual + LN + attention + previous MoE), poisoning everything, and Adam writes NaN into all parameters.

This is the **production failure mode** with PR #45621's sentinel-skip pattern: under EP=8, ~87.5 % of slots are sentinels, almost every token has at least one sentinel slot → almost every row of `dH` is NaN.

---

## 4. `masked_fill` forward and backward

### Math

The forward `y = x.masked_fill(M, c)` is:

$$y[i] \;=\; \begin{cases} c & \text{if } M[i] = \text{True} \\ x[i] & \text{otherwise} \end{cases}$$

Differentiating w.r.t. `x[i]`:

$$\frac{\partial y[i]}{\partial x[i]} \;=\; \begin{cases} 0 & \text{if } M[i] = \text{True} \\ 1 & \text{otherwise} \end{cases}$$

Chain rule for the backward:

$$dx[i] \;=\; \frac{\partial L}{\partial y[i]} \cdot \frac{\partial y[i]}{\partial x[i]} \;=\; \begin{cases} 0 & \text{if } M[i] = \text{True} \\ dy[i] & \text{otherwise} \end{cases}$$

In compact form: **`dx = dy.masked_fill(M, 0)`**. Crucially, **the value of `dy[i]` at masked positions is irrelevant** — backward overwrites it with `0`.

### Torch

```python
def masked_fill_backward(dy, mask):
    return dy.masked_fill(mask, 0.0)
```

The autograd implementation is exactly this. PyTorch source: `aten::masked_fill`'s backward is a `where(mask, 0, dy)` essentially.

### Tiny example: `masked_fill` is a NaN absorber in backward

```python
x = torch.tensor([1., 2., 3., 4.], requires_grad=True)
mask = torch.tensor([False, True, False, True])
y = x.masked_fill(mask, 0.0)
print(y)
# tensor([1., 0., 3., 0.], grad_fn=<MaskedFillBackward0>)

# Send NaN gradient through the masked positions:
dy = torch.tensor([10., float('nan'), 30., float('nan')])
y.backward(dy)
print(x.grad)
# tensor([10.,  0., 30.,  0.])   ← sentinel positions cleanly zeroed
```

**The NaN never reaches `x.grad`.** `masked_fill`'s backward is a "firewall" that doesn't care about the value of incoming gradients at masked positions — it sets them to 0 unconditionally.

This is the property the fix exploits.

---

## 5. The fix in the autograd graph

The wrapper does:

```python
selected = H[idx]                                   # gather
selected = selected.masked_fill(sentinel, 0.0)      # ← the firewall
out = grouped_mm(selected, W, offsets)              # kernel may NaN d_selected at sentinel rows
```

The autograd graph (forward direction):

```
H ──gather──> selected ──masked_fill──> selected_zero ──grouped_mm──> out
```

Backward direction (gradient flows right-to-left):

```
d_out ──grouped_mm.bwd──> d_selected_zero ──masked_fill.bwd──> d_selected ──gather.bwd──> dH
```

Walking through what happens at sentinel positions:

| node                  | sentinel row value         | how it got there                          |
|-----------------------|----------------------------|-------------------------------------------|
| `d_out` from above    | could be anything          |                                           |
| `d_selected_zero`     | **NaN** (kernel uninit)    | `_grouped_mm` skips writing past offsets  |
| `d_selected`          | **0** ← masked_fill bwd    | `dy.masked_fill(M, 0)` overwrites NaN     |
| `dH` after gather bwd | finite (only real contribs)| `index_add_` adds 0 from sentinel rows     |

The critical line is `d_selected[sentinel] = 0` — **regardless of what `d_selected_zero[sentinel]` was**, masked_fill's backward sets it to 0. NaN dies there.

### Tiny example: the full pipeline

```python
import torch

T, S, D = 3, 6, 2

H = torch.tensor([[1., 2.],
                  [3., 4.],
                  [5., 6.]], requires_grad=True)
idx = torch.tensor([0, 0, 1, 1, 2, 2])
sentinel = torch.tensor([False, True, False, True, False, True])

selected = H[idx]                                         # (S, D)
selected_zero = selected.masked_fill(sentinel.unsqueeze(-1), 0.0)
# selected_zero forces sentinel rows to zero in forward (we don't actually run grouped_mm here;
# instead we'll inject NaN gradients at sentinel rows of d_selected_zero to simulate the kernel)

# Some downstream loss-like reduction
loss = selected_zero.sum()

# Manually inject what the kernel's broken backward would produce: NaN at sentinel rows of d_selected_zero.
# We do this by creating a custom grad and calling backward on it, but PyTorch normally computes d_selected_zero
# from loss.sum's backward (which gives all-ones). To inject NaN, we instead call .backward(grad) on selected_zero
# directly with a hand-crafted grad.

selected_zero.backward(torch.tensor([[1., 1.],
                                     [float('nan'), float('nan')],   # sentinel
                                     [1., 1.],
                                     [float('nan'), float('nan')],   # sentinel
                                     [1., 1.],
                                     [float('nan'), float('nan')]])) # sentinel

print("dH (should be all finite, NO NaN):")
print(H.grad)
# tensor([[1., 1.],   ← only the real (non-sentinel) gradients summed in
#         [1., 1.],
#         [1., 1.]])
```

If you remove the `masked_fill` line, the same `backward` call gives `dH = [[nan,nan],[nan,nan],[nan,nan]]`. The masked_fill in the middle of the graph is the entire firewall.

---

## 6. Why the kernel leaves uninitialized rows in the first place

`torch._grouped_mm` is a CUDA op designed to do batch-of-matmuls efficiently. It takes:
- `input: (S, K)` rows of input vectors
- `W: (E, K, M)` per-expert weights
- `offsets: (E,)` cumulative-sum boundaries

For each expert `e`, the kernel does:

```
output[offsets[e-1] : offsets[e], :]  =  input[offsets[e-1] : offsets[e], :]  @  W[e]
```

It iterates `e ∈ [0, E)` and writes the slice `[0, offsets[-1])` of `output`. **Rows past `offsets[-1]` are never touched.** They came from `torch.empty_like(...)` which doesn't zero-init. In practice that memory is reused from the CUDA caching allocator pool, and in a real training loop the pool is full of stale floats from prior tensors — including NaNs from intermediate ops.

The same is true in backward:

```
d_input[offsets[e-1] : offsets[e], :]  =  d_output[offsets[e-1] : offsets[e], :]  @  W[e].T
```

`d_input[offsets[-1] : S]` is allocated from `torch.empty_like(input)` and never written.

The proper fix for the kernel is one line: replace `torch.empty_like(input)` with `torch.zeros_like(input)`. Then `d_input[sentinel] = 0` cleanly and the wrapper doesn't need a firewall. (See Comment 2 of the upstream PR.)

---

## 7. Why two `masked_fill`s per `grouped_mm` (input + output)?

The wrapper does:

```python
selected = selected.masked_fill(sentinel, 0.0)   # (a) before
out = grouped_mm(selected, W, offsets)
out = out.masked_fill(sentinel, 0.0)             # (b) after
```

(a) covers backward: `d_selected[sentinel] = 0` via masked_fill backward. Closes the leak we walked through.

(b) covers forward: `out[sentinel]` is uninitialized in forward (same kernel issue, on the forward output). Without (b), `out[sentinel]` is garbage which propagates into SwiGLU(garbage) = garbage and into the next grouped_mm's input → output → multiply → loss = potentially NaN forward too.

Both directions need protection. They use the *same* mechanism (`masked_fill`) because the property of "zero at masked positions" is what we need on both ends.

---

## 8. Hands-on exercises

### Exercise 1: prove `masked_fill` is a NaN firewall

```python
x = torch.randn(8, requires_grad=True)
mask = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.bool)
y = x.masked_fill(mask, 0.0)
y.backward(torch.tensor([float('nan'), 1., float('nan'), 1.,
                         float('nan'), 1., float('nan'), 1.]))
assert not torch.isnan(x.grad).any()
print(x.grad)   # NaN positions are 0, others passed through
```

### Exercise 2: count how many rows of `dH` get poisoned for varying sentinel rates

```python
def poison_count(p_sentinel, T=1024, K=8):
    S = T * K
    sentinel = torch.rand(S) < p_sentinel
    idx = torch.arange(S) // K  # token id of each slot
    dY = torch.zeros(S, 1)
    dY[sentinel] = float('nan')
    dY[~sentinel] = 0.5
    H = torch.zeros(T, 1, requires_grad=True)
    Y = H[idx]
    Y.backward(dY)
    return torch.isnan(H.grad).sum().item(), T

for p in [0.1, 0.5, 0.875, 0.94]:
    n, T = poison_count(p)
    print(f"p_sentinel={p:.3f}  poisoned tokens = {n}/{T}  ({100*n/T:.1f}%)")
```

You'll see that already at `p_sentinel = 0.5`, almost all tokens are poisoned, because each token has many slots and only one bad slot suffices.

### Exercise 3: write the fix yourself

Take Exercise 2's `poison_count`, insert a `masked_fill(sentinel, 0)` between the gather and the backward, and confirm `H.grad` is all finite.

---

## 9. Summary table

| layer of the chain                                    | math                                                  | torch op             |
|-------------------------------------------------------|-------------------------------------------------------|----------------------|
| `Y = H[idx]`                                          | `Y[r] = H[idx[r]]`                                    | fancy indexing       |
| backward of fancy indexing                            | `dH[t] = Σ_{r: idx[r]=t} dY[r]`                       | `index_add_`         |
| `+=` semantics                                        | `real + NaN = NaN` (IEEE 754)                         | follows IEEE         |
| `y = x.masked_fill(M, c)`                             | `y[i] = c if M[i] else x[i]`                          | `masked_fill`        |
| backward of masked_fill                               | `dx[i] = 0 if M[i] else dy[i]` (independent of `dy[i]` at masked positions!) | `dy.masked_fill(M, 0)` |
| `_grouped_mm` forward output past `offsets[-1]`       | uninitialized                                         | `torch.empty_like`   |
| `_grouped_mm` backward `d_input` past `offsets[-1]`   | uninitialized                                         | `torch.empty_like`   |

The fix:
> Place `masked_fill(sentinel, 0)` on **both sides** of every `_grouped_mm` call. Forward correctness is given by the post-mask. Backward correctness is given by the pre-mask, because `masked_fill`'s backward unconditionally zeros gradient at masked positions, absorbing whatever NaN the kernel wrote into `d_input` at sentinel rows before it reaches the upstream `index_add_`.

That's the entire story. Five lines of `masked_fill` instead of a custom autograd Function or kernel patch.
