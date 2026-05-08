# Post-Training Toolkit Integration

[Post-Training Toolkit](https://github.com/microsoft/post-training-toolkit) is a diagnostic and observability layer for RLHF training runs. Add one callback to any TRL trainer and get **auto-metrics**, **crash postmortems**, and **literature-backed heuristics**—without writing glue code.

It was built to operationalize the debugging patterns we found most useful when running post-training at scale.

## Usage

1. First, install Post-Training Toolkit:

```bash
pip install post-training-toolkit
```

2. Add one callback to your trainer. That's it!

<hfoptions id="trainer">
<hfoption id="DPO">

```python
from post_training_toolkit import DiagnosticsCallback
from trl import DPOTrainer

trainer = DPOTrainer(
    model=model,
    args=training_args,
    callbacks=[DiagnosticsCallback()],  # ← Just add this
    ...
)
trainer.train()
```

</hfoption>
<hfoption id="PPO">

```python
from post_training_toolkit import DiagnosticsCallback
from trl.experimental.ppo import PPOTrainer

trainer = PPOTrainer(
    model=model,
    args=training_args,
    callbacks=[DiagnosticsCallback()],  # ← Just add this
    ...
)
trainer.train()
```

</hfoption>
<hfoption id="SFT">

```python
from post_training_toolkit import DiagnosticsCallback
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    args=training_args,
    callbacks=[DiagnosticsCallback()],  # ← Just add this
    ...
)
trainer.train()
```

</hfoption>
<hfoption id="ORPO">

```python
from post_training_toolkit import DiagnosticsCallback
from trl.experimental.orpo import ORPOTrainer

trainer = ORPOTrainer(
    model=model,
    args=training_args,
    callbacks=[DiagnosticsCallback()],  # ← Just add this
    ...
)
trainer.train()
```

</hfoption>
<hfoption id="KTO">

```python
from post_training_toolkit import DiagnosticsCallback
from trl import KTOTrainer

trainer = KTOTrainer(
    model=model,
    args=training_args,
    callbacks=[DiagnosticsCallback()],  # ← Just add this
    ...
)
trainer.train()
```

</hfoption>
<hfoption id="CPO">

```python
from post_training_toolkit import DiagnosticsCallback
from trl.experimental.cpo import CPOTrainer

trainer = CPOTrainer(
    model=model,
    args=training_args,
    callbacks=[DiagnosticsCallback()],  # ← Just add this
    ...
)
trainer.train()
```

</hfoption>
<hfoption id="GRPO">

```python
from post_training_toolkit import DiagnosticsCallback
from trl import GRPOTrainer

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    callbacks=[DiagnosticsCallback()],  # ← Just add this
    ...
)
trainer.train()
```

</hfoption>
</hfoptions>

## What You Get

**Example output:**
```text
[HIGH] DPO loss stuck at ~0.693 (random chance). Model may not be learning preferences.
       Ref: Rafailov et al. (2023) 'DPO', Section 4.2

[RECOMMENDED] Increase learning rate 2-5x, check data quality, or reduce beta.
```

## Example Demo

See a full working example with auto-stop in action:

📂 **[demo/live_demo.ipynb](https://github.com/microsoft/post-training-toolkit/blob/main/demo/notebooks/demo_live_output.ipynb)**

📂 **[demo/scripts/custom_heuristic.py](https://github.com/microsoft/post-training-toolkit/blob/main/demo/scripts/custom_heuristic_demo.py)**


### 1. Auto-Metrics
The callback automatically captures algorithm-specific metrics, backed by the latest research and industry push:

| Trainer | Key Metrics Captured |
|---------|---------------------|
| **DPO** | loss, win_rate, reward_margin, logps_chosen/rejected |
| **PPO** | policy_loss, value_loss, entropy, clip_fraction, KL |
| **GRPO** | group rewards, advantages, policy loss, KL |
| **SFT** | loss, perplexity, accuracy |
| **ORPO** | sft_loss, odds_ratio_loss, log_odds_ratio |
| **KTO** | kl, logps for desirable/undesirable |


### 2. Crash Postmortems
If training crashes or gets interrupted, you get a `postmortem.json` with full context:

```json
{
  "exit_reason": "exception",
  "last_step": 847,
  "timestamp": "2025-12-17T19:26:04Z",
  "final_metrics": {"dpo_loss": 0.693, "win_rate": 0.52}
}
```

No more "what step did it die on?"

### 3. Auto-Stop on Critical Issues

Enable automatic training termination when critical issues are detected:

```python
callback = DiagnosticsCallback(stop_on_critical=True)
```

## Distributed Training
Works automatically with multi-GPU setups. Zero configuration needed:

```bash
accelerate launch --num_processes 8 train.py
```

Automatically detects stragglers, aggregates metrics across ranks, and tracks memory balance.
