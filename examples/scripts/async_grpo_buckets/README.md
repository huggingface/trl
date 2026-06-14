# Disaggregated async GRPO with bucket weight sync (`async_grpo_buckets.py`)

Train a policy on your **local GPU** while a **remote vLLM HF Space** does generation — the two never share a NCCL
group. They stay in sync through an **HF Storage Bucket**: after each optimizer step the trainer uploads only the bf16
weights that changed (a sparse patch, recovered by inverting the AdamW step — no snapshot), and the remote vLLM applies
it in place. A full **anchor** is sent every N syncs to bound drift.

This is the power of disaggregation: **training and inference scale and live independently.** Put the trainer wherever
your training GPUs are, serve generation from an autoscaling Space (or many), and connect any environment server — all
glued together by a bucket and plain HTTP.

```
        ┌──────────────────────────┐     sparse patches / anchors      ┌───────────────────────────┐
        │  Local trainer (1 GPU)   │ ───────────────────────────────▶  │   HF Storage Bucket        │
        │  AsyncGRPOTrainer        │                                   │   anchors/ + deltas/       │
        │  + rollout worker        │ ◀───────────────────────────────  └───────────────────────────┘
        └──────────┬───────────────┘            apply in place                      ▲
                   │ /v1/completions (HTTP)                                          │ fetch
                   ▼                                                                 │
        ┌──────────────────────────┐                                   ┌───────────────────────────┐
        │  vLLM HF Space (GPU)      │ ◀──────────────────────────────  │  HFBucketWorkerExtension  │
        │  serves generation        │                                   │  (hf_bucket backend)      │
        └──────────┬───────────────┘                                   └───────────────────────────┘
                   │ tool calls (HTTP)
                   ▼
        ┌──────────────────────────┐
        │  Wordle env HF Space      │   (no GPU; public one at openenv-wordle.hf.space)
        └──────────────────────────┘
```

Files in this directory:

- `async_grpo_buckets.py` — the local trainer (AsyncGRPO + Wordle env, `weight_sync_backend="bucket"`).
- `vllm_space/` — Dockerfile + README to deploy the **vLLM inference Space** (GPU).
- `wordle_space/` — Dockerfile + README to deploy your own **Wordle environment Space** (optional; a public one exists).

## Prerequisites

```sh
pip install "trl @ git+https://github.com/huggingface/trl.git@delta-weight-sync-v3"
pip install "openenv-textarena @ git+https://huggingface.co/spaces/openenv/wordle"  # the Wordle env client
hf auth login   # needs write access to create the bucket + (for Option 1) deploy Spaces
```

The vLLM side needs a build with sparse weight transfer (vllm-project/vllm#40096); the Space Dockerfile installs it
from the nightly index. Locally, install the same nightly (see the repo's `dev_delta_v2/INSTALL.md`).

### Step 1 — deploy the vLLM inference Space (GPU)

```sh
# Create the Space (l4 GPU, Docker SDK). HF_TOKEN lets the Space read the bucket.
hf repos create <your-username>/vllm-wordle-inference \
    --repo-type space --space-sdk docker

hf upload <your-username>/vllm-wordle-inference \
    examples/scripts/async_grpo_buckets/vllm_space/ . --repo-type space

# Set the GPU + secrets/vars in the Space settings (or via the CLI / web UI):
#   hardware: l4x1 ;  secret HF_TOKEN=<token> ;  the Dockerfile already sets VLLM_SERVER_DEV_MODE=1
```

Wait until `https://<your-username>-vllm-wordle-inference.hf.space/health` returns 200 (first build pulls the image and
loads the model — a few minutes).

### Step 2 — (optional) deploy your own Wordle env Space

A public env runs at `https://openenv-wordle.hf.space`. To run your own (higher concurrency), deploy `wordle_space/`
the same way and pass its URL via `--env-url`.

### Step 3 — train locally (1 GPU)

```sh
CUDA_VISIBLE_DEVICES=0 python examples/scripts/async_grpo_buckets/async_grpo_buckets.py \
    --model Qwen/Qwen3-1.7B \
    --vllm-server-url https://<your-username>-vllm-wordle-inference.hf.space \
    --env-url https://openenv-wordle.hf.space \
    --weight-sync-bucket-id <your-username>/wordle-deltas
```

The bucket (`<your-username>/wordle-deltas`) is created automatically on the first sync.

## Notes

- **Bucket vs NCCL.** Bucket sync works across hosts/Spaces (data plane = HF Hub, control plane = HTTP), at the cost of
  object-storage latency (~seconds/sync). On a single node where the trainer and vLLM share NVLink, the default
  `weight_sync_backend="nccl"` is ~100× faster — use the bucket backend specifically for the disaggregated/cross-host
  case this example demonstrates.
- **Anchors.** `--weight-sync-anchor-interval N` uploads a full checkpoint every N syncs (sparse deltas in between) to
  bound drift from any missed bits. Lower N = more robust, larger uploads.
- **Key flags must match** between the example and the Space: `Qwen/Qwen3-1.7B`, the `hf_bucket` backend, and the
  `HFBucketWorkerExtension`.
