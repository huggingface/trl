"""Compiled PyTorch scan baseline for the benchmark's shared-prefix star trees.

Compile eight chunks per region so 16K/32K/64K do not require enormous unrolled compilation graphs. Siblings run in
one batch. Python launches the regions; timing includes these launches. This is the FP32 reference implementation,
not FLA's fused low-precision chunk algebra. The benchmark therefore compares complete implementations, not just scans.
"""

import torch

from trl.kernels.tree_gated_delta_rule import _chunk_algebra, _chunk_output


@torch.compile(fullgraph=True, dynamic=True)
def _scan_region(kbar, w, u, decay, state):
    states, values = [], []
    for i in range(kbar.shape[0]):
        states.append(state)
        value = u[i] - w[i] @ state
        state = decay[i, ..., None, None] * state + kbar[i].transpose(-1, -2) @ value
        values.append(value)
    return state, torch.stack(states), torch.stack(values)


def torch_tree_gdn(q, k, v, g, beta, plan, prefix, branches):
    with torch.autocast(device_type="cuda", enabled=False):
        qbar, attention, kbar, w, u, decay = _chunk_algebra(
            q,
            k,
            v,
            g,
            beta,
            plan.chunk_tokens,
            q.shape[-1] ** -0.5,
            True,
        )
        prefix_chunks = (prefix + 63) // 64
        outputs_h, outputs_v = [], []
        state = w.new_zeros((1, v.shape[2], k.shape[-1], v.shape[-1]))
        for start, end, count in [(0, prefix_chunks, 1), (prefix_chunks, w.shape[0], branches)]:
            state = state.expand(count, -1, -1, -1).contiguous()
            xs = [x[start:end].reshape(count, -1, *x.shape[1:]).transpose(0, 1) for x in (kbar, w, u, decay)]
            hs, vs = [], []
            for offset in range(0, xs[0].shape[0], 8):
                state, h, value = _scan_region(*(x[offset : offset + 8] for x in xs), state)
                hs.append(h)
                vs.append(value)
            outputs_h.append(torch.cat(hs).transpose(0, 1).flatten(0, 1))
            outputs_v.append(torch.cat(vs).transpose(0, 1).flatten(0, 1))
        return _chunk_output(qbar, attention, torch.cat(outputs_h), torch.cat(outputs_v), plan.output_tokens, v.dtype)
