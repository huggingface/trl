# Migrating from TRL v0 to v1

This guide covers the breaking changes introduced in TRL v1 and how to update your code. Most structural changes (trainers moved to experimental, removed model classes, etc.) already shipped in v0.29 — if you're already on v0.29, this migration is minimal.

## Changed defaults

| Config | Parameter | v0 default | v1 default | Action needed |
|---|---|---|---|---|
| `GRPOConfig` | `vllm_mode` | `"server"` | `"colocate"` | If you use `use_vllm=True` without specifying `vllm_mode`, vLLM will now run in the same process instead of connecting to a separate server. Set `vllm_mode="server"` explicitly if you rely on server mode. |
| `RLOOConfig` | `vllm_mode` | `"server"` | `"colocate"` | Same as above. |

## Already changed in v0.29

The following changes were introduced in v0.29 and are **not new in v1**. They are listed here for completeness if you are migrating from an earlier version.

<details>
<summary>Trainers moved to experimental</summary>

Several trainers were moved from the stable API to `trl.experimental`. They are no longer importable from `trl` directly (except KTO, which still has a compatibility shim with a deprecation warning).

| Trainer | New import |
|---|---|
| PPO | `from trl.experimental.ppo import PPOTrainer, PPOConfig` |
| CPO | `from trl.experimental.cpo import CPOTrainer, CPOConfig` |
| BCO | `from trl.experimental.bco import BCOTrainer, BCOConfig` |
| ORPO | `from trl.experimental.orpo import ORPOTrainer, ORPOConfig` |
| XPO | `from trl.experimental.xpo import XPOTrainer, XPOConfig` |
| Online DPO | `from trl.experimental.online_dpo import OnlineDPOTrainer, OnlineDPOConfig` |
| GKD | `from trl.experimental.gkd import GKDTrainer, GKDConfig` |
| Nash-MD | `from trl.experimental.nash_md import NashMDTrainer, NashMDConfig` |
| PRM | `from trl.experimental.prm import PRMTrainer, PRMConfig` |
| KTO | `from trl.experimental.kto import KTOTrainer, KTOConfig` |

</details>

<details>
<summary>Removed model classes</summary>

| Class | New location |
|---|---|
| `AutoModelForCausalLMWithValueHead` | `trl.experimental.ppo` |
| `AutoModelForSeq2SeqLMWithValueHead` | `trl.experimental.ppo` |
| `PreTrainedModelWrapper` | `trl.experimental.ppo` |

</details>

<details>
<summary>Removed callbacks and utilities</summary>

| What | New location |
|---|---|
| `WinRateCallback` | `trl.experimental.winrate_callback` |
| Judges | `trl.experimental.judges` |
| `peft_module_casting_to_bf16` | `trl.experimental.utils` |
| `FDivergenceType` enum | Removed. Use string values (`"reverse_kl"`, `"js_divergence"`, `"alpha_divergence"`) directly. |

</details>
