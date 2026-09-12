# UP

In the paper [UP: Unbounded Positive Asymmetric Optimization for Breaking the Exploration-Stability Dilemma](https://huggingface.co/papers/2607.06987), the authors propose a GRPO variant that routes tokens asymmetrically on the sign of the advantage. Clipping stabilizes training but truncates the update budget of correct yet low-confidence tokens. UP removes the clip only where it hurts exploration: for positive advantages the importance sampling ratio is replaced by the self-anchored ratio `πθ / sg(πθ)`, whose value is exactly 1 and whose gradient is the unclipped REINFORCE gradient, independent of the old policy. Non-positive advantages keep the standard clipped surrogate as a safeguard.

To use UP, you can use the [`UPTrainer`] class in `trl.experimental.up`.

## Usage

```python
from trl.experimental.up import UPConfig, UPTrainer

training_args = UPConfig(
    epsilon=0.2,  # lower clip bound; only applies to the non-positive-advantage branch
    beta=0.0,  # the paper's UP-DAPO setup, reported at 14B
)
trainer = UPTrainer(
    model="Qwen/Qwen3-0.6B",
    reward_funcs=...,
    train_dataset=...,
    args=training_args,
)
trainer.train()
```

`epsilon_high` has no effect with the default `delta=None`, and more generally whenever `delta >= 1 + epsilon_high`: positive advantages bypass clipping entirely, and for non-positive ones the upper bound is dominated. Only a `delta` set below `1 + epsilon_high` makes it bind. Table A1 of the paper lists no `ε_high` for UP-DAPO accordingly. `epsilon` and the optional two-sided cap `delta` still apply to the non-positive branch.

Aggregation follows DAPO's global active-token normalization, matching the paper's UP-DAPO instantiation.

Setting `importance_sampling_level="sequence"` changes only the non-positive branch, which then uses the sequence-level ratio of the paper's UP-GSPO (Eq. 16). The positive branch is unaffected: the global active-token normalization exactly cancels the sequence-level length normalization, so its loss and gradient are identical to the token-level case. Aggregation also stays token-normalized rather than averaging per-sequence losses as the paper's UP-GSPO does.

The paper's UP-DAPO results are at 14B with `beta=0.0`. At smaller scale, consider keeping a KL term (`beta > 0`), as the paper's UP-GRPO variant does.

## UPTrainer

[[autodoc]] experimental.up.UPTrainer
    - train
    - save_model
    - push_to_hub

## UPConfig

[[autodoc]] experimental.up.UPConfig
