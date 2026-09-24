---
library_name: kernels
{% if license %}license: {{ license }}
{% endif %}---

This is the repository card of {{ repo_id }} that has been pushed on the Hub. It was built to be used with the
[`kernels` library](https://github.com/huggingface/kernels).

## How to use

```python
from kernels import get_kernel

trl_losses = get_kernel(
    "{{ repo_id }}",
    version={{version}},
    trust_remote_code=True,
)
logprobs, entropy = trl_losses.selective_log_softmax_and_entropy(logits, index)
```

## Available functions

- `selective_log_softmax_and_entropy`

## Source code

The source is maintained in the
[`huggingface/trl`](https://github.com/huggingface/trl/tree/main/kernels) repository.
