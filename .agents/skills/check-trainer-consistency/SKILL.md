---
name: check-trainer-consistency
description: Keep duplicated trainer code aligned across TRL trainers. Use when modifying or reviewing code in any trainer (GRPO, RLOO, SFT, DPO, ...) that also exists in sibling trainers, e.g. generation paths, reward computation, metric logging, or weight syncing.
---

# Check trainer consistency

Trainers in TRL are **self-contained by design**. Shared logic (vLLM generation paths, `_get_per_token_logps_and_entropies`, `_calculate_rewards`, `_prepare_inputs`, metric logging, weight syncing) is deliberately duplicated across trainers instead of being abstracted into a base class, so each trainer stays readable and evolvable in isolation.

The tradeoff: duplication is accepted, but **consistency is mandatory**.

## Rules for duplicated blocks

- Same variable names (`self._last_loaded_step`, `self._metrics[mode]`, ...).
- Same control-flow structure (if/elif/else branches in the same order).
- Same comments, word-for-word when the logic is identical.
- Divergences only where the trainer's semantics require it (e.g. GRPO extracts logprobs from vLLM, RLOO discards them).

**Consistency over correctness.** When duplicating code, reproduce it exactly — even if you believe the original has a bug. Do not silently fix the issue in your copy: keep it consistent and report the problem so it can be fixed across all trainers in one dedicated PR. A consistently-wrong codebase can be fixed in a single sweep; an inconsistent one cannot.

## When modifying duplicated code

1. Identify the duplicated block you are changing.
2. Find every copy. Grep a distinctive line of the block across the main and experimental trainers:
   ```sh
   grep -rn "self._last_loaded_step" trl/trainer/ trl/experimental/
   ```
3. Apply the same change to every copy. A fix in GRPO usually implies the same fix in RLOO and vice versa. Not propagating a change is a bug.
4. Verify the copies stayed aligned by diffing the corresponding regions, e.g.:
   ```sh
   diff <(sed -n '/def _generate_single_turn/,/def /p' trl/trainer/grpo_trainer.py) \
        <(sed -n '/def _generate_single_turn/,/def /p' trl/trainer/rloo_trainer.py)
   ```
   Remaining diffs must all be semantic divergences, not drift.

## When reviewing

If a PR touches duplicated logic, check that all copies were updated consistently. The most common mistake is fixing one trainer and forgetting the others.

## Scope

Main code (`trl/trainer/`) must stay stable and consistent. Experimental code (`trl/experimental/`) may lag; small non-invasive alignment improvements are welcome, large refactors are not.
