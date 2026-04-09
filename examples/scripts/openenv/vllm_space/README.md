---
title: vLLM Wordle Inference
emoji: 🎮
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
hardware: l4
---

vLLM server with DeltaWorkerExtension for async GRPO training.

Serves Qwen/Qwen3-1.7B with delta weight sync via HF Hub buckets.
Used by `examples/scripts/openenv/async_wordle.py` in the TRL repo.
