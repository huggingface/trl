# Paper index

This file tracks research papers whose methods are implemented in TRL. Each
subsection links the paper, describes the contribution, and lists the relevant
trainer(s) / config(s).

---

## X-Token: Cross-Tokenizer Knowledge Distillation

**Paper**: [X-Token: Rethinking Experience in Alignment Distillation](https://huggingface.co/papers/2605.21699)

**Abstract**: X-Token extends the GOLD cross-tokenizer distillation framework
with two new KD loss formulations that handle the vocabulary mismatch between
student and teacher when their tokenizers differ.

**Variants**

| Variant | Config value | Description |
|---------|-------------|-------------|
| P-KL | `xtoken_loss_type="p_kl"` | Projects the full student distribution to the teacher vocab space via a sparse projection matrix W and computes forward KL. Recovers signal for tokens in the uncommon set (e.g. multi-digit numerals that split differently across tokenizer families). Implements Eq. (2) of the paper. |
| H-KL | `xtoken_loss_type="h_kl"` | Retains the GOLD hybrid structure but builds the common set via the top-1 mapping under W (threshold ≥ 0.6) instead of strict string equality. Applies forward KL on common tokens and sorted-L1 on uncommon tokens. Implements Eq. (3-4) of the paper. |

**Key features**
- Sparse projection matrix W ∈ ℝ^{V_s × V_t} precomputed offline and loaded lazily per worker process.
- FP32 sparse matmul (`_Fp32SparseMM`) that ignores surrounding BF16 autocast — required because PyTorch has no BF16 sparse-mm kernel.
- T² gradient scaling (Hinton 2015) for temperature-independent gradient magnitudes.
- Optional stop-gradient dynamic CE/KD balancing (`xtoken_dynamic_scaling=True`, paper Eq. 7).
- Token-span alignment reuses TRL's existing byte-offset method (`ULDLoss._align_by_byte_offsets`).

**Relevant files**
- `trl/experimental/gold/gold_config.py` — `GOLDConfig` fields prefixed `xtoken_*`
- `trl/experimental/gold/gold_trainer.py` — `XTokenLoss` class, `_load_sparse_projection_matrix`, `_load_exact_token_map`, `_Fp32SparseMM`

**Reference implementation**: [NVIDIA-NeMo/RL PR #2508](https://github.com/NVIDIA-NeMo/RL/pull/2508), [PR #2757](https://github.com/NVIDIA-NeMo/RL/pull/2757), [PR #2797](https://github.com/NVIDIA-NeMo/RL/pull/2797)
