---
tags:
  - kernel
  - triton
  - trl
---

# TRL loss kernels

Fused Triton kernels for operations after a language model's decoder. The first operation computes selected
log-probabilities and Shannon entropy in one pass over materialized logits, with an autograd implementation for both
outputs.

The source is maintained in
[`huggingface/trl`](https://github.com/huggingface/trl/tree/main/kernels).

```python
from kernels import get_kernel

trl_losses = get_kernel(
    "trl-lib/trl-losses",
    version=0,
    trust_remote_code=["trl-lib/trl-losses"],
)
logprobs, entropy = trl_losses.selective_log_softmax_and_entropy(logits, token_ids)
```

The operation accepts fp32, fp16, or bf16 logits shaped `[tokens, vocab]` or `[batch, tokens, vocab]`. Outputs are
accumulated in fp32. CUDA, ROCm, and XPU use the same backend-neutral Triton source.

## Benchmarks

In the prototype benchmark reported in [TRL #7244](https://github.com/huggingface/trl/issues/7244), the fused forward
takes 0.89 ms on an H100 with bf16 logits shaped `[8, 1024, 151936]`. The equivalent separate PyTorch
selective-log-softmax and entropy paths take 12.8 ms and materialize additional full-vocabulary temporaries.
