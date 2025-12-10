# GSPO-token

In the paper [Group Sequence Policy Optimization](https://huggingface.co/papers/2507.18071), the authors propose a token-level objective variant to GSPO, called GSPO-token. To use GSPO-token, you can use the `GRPOTrainer` class in `trl.experimental.gspo_token`.

## Usage

```python
from trl.experimental.gspo_token import GRPOTrainer
from trl import GRPOConfig

training_args = GRPOConfig(
    importance_sampling_level="sequence_token",
    ...
)
```

> [!WARNING]
> To leverage GSPO-token, the user will need to provide the per-token advantage  \\( \hat{A_{i,t}} \\) for each token  \\( t \\) in the sequence  \\( i \\) (i.e., make  \\( \hat{A_{i,t}} \\) varies with  \\( t \\)—which isn't the case here,  \\( \hat{A_{i,t}}=\hat{A_{i}} \\)). Otherwise, GSPO-Token gradient is just equivalent to the original GSPO implementation.

## GRPOTrainer

[[autodoc]] experimental.gspo_token.GRPOTrainer
    - train
    - save_model
    - push_to_hub
