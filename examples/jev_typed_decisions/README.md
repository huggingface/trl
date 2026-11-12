# JEV decisions

This example collator adapts Transformers' native JEV prompt and packed option targets to `DecisionTrainer`.
Requires [Transformers #50586](https://github.com/huggingface/transformers/pull/50586).

Use `DataCollatorForJev` from `collator.py` with the model's tokenizer and `decision_config`. Each training row has
one question in the standard `state` / `questions` request, plus `target_index`, `target_probs`, `p_true`, or `score`.
`target_probs` includes a zero null entry followed by the option probabilities; the collator removes that entry for
the native loss. Nullable decisions are unsupported.

Set both `per_device_train_batch_size` and `per_device_eval_batch_size` to `1`; use gradient accumulation for more
decisions per update. Set `gradient_checkpointing=False` for the current JEV model. Targets can be prepared once with `dataset.map(collator.tokenize)`.
The model computes its own KL and ordinal RPS objective; the trainer does not reimplement it.
