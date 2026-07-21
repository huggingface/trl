# Switch tests, docs, and example to prompt-only

On-policy distillation only needs prompts (the student generates completions), so the training surface should use prompt-only datasets. Now unblocked by the `num_items` fix (#6478), which stops prompt-only batches from NaN-ing.

- **Tests:** `conversational_language_modeling` → `conversational_prompt_only` in the end-to-end training/init tests. Kept on messages: the deprecation test and `_make_local_trainer` (feeds `test_loss_normalizes`, which reads completion tokens straight from the collator — switches to prompt-only once generation replaces the collator).
- **Docs + example:** quick-start and example script now use the `prompt` column (`trl-lib/ultrafeedback-prompt`, a conversational prompt-only dataset). Also removed the dead `lmbda` / off-policy references (the param was already removed from the config).

Verified: `pytest tests/experimental/test_distillation_trainer.py` — 40 passed.
