# CLM on typed decisions

Train new CLM projection heads on a frozen SmolLM2-135M encoder using
[`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions).
Requires [Transformers #50246](https://github.com/huggingface/transformers/pull/50246).

```bash
python examples/clm_typed_decisions/clm_typed_decisions.py --config examples/clm_typed_decisions/config.yaml
```

For a short GPU check:

```bash
python examples/clm_typed_decisions/clm_typed_decisions.py \
    --config examples/clm_typed_decisions/config.yaml \
    --max_train_samples 32 --max_eval_samples 8 --max_steps 20 --max_length 256
```

The configuration uses three-layer heads with width 1,536 and projection dimension 512,
AdamW at `5e-4`, zero weight decay, and a cosine OneCycle schedule with 10% warmup.
Training runs for up to 20 epochs, with early stopping after five evaluations without improvement.
The per-device batch size is eight to keep the example small.

Validation holds out 10% of source-row IDs before expanding rows into questions. The published
test split remains available for final benchmarking. Inputs are tokenized once, retaining their
last 2,047 tokens. `max_train_samples` and `max_eval_samples` limit source rows.

Each batch deduplicates candidates and trains with bidirectional InfoNCE against soft target
distributions. `eval_accuracy` and `eval_candidate_loss` use each question's own candidates;
`eval_loss` is the contrastive objective over the batch's candidate pool. Negatives are local to
each device; gradient accumulation does not expand that pool.

The script saves a complete Transformers checkpoint. Reload it with `pipeline("decision", model="output/clm-typed-decisions")`.
