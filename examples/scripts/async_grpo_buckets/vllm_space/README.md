---
title: vLLM Wordle Inference
emoji: 🎮
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
hardware: l4
---

vLLM inference server for **disaggregated** async GRPO training.

Serves `Qwen/Qwen3-1.7B` and keeps in sync with a remote trainer via **bucket weight sync**: the trainer uploads
the changed bf16 weights as sparse patches to an HF Storage Bucket, and this Space applies them in place using the
`hf_bucket` weight-transfer backend (registered by the `HFBucketWorkerExtension`, served with `--model-impl
transformers` + `VLLM_USE_V2_MODEL_RUNNER=0` so the in-place sparse apply works).

Used by `examples/scripts/async_grpo_buckets/async_grpo_buckets.py` in the TRL repo. See
`examples/scripts/async_grpo_buckets/README.md` for the end-to-end deploy + run guide.
