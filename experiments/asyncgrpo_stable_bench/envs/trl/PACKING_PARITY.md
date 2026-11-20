# TRL packed-vs-alone log-prob parity on Qwen3.5-2B

TRL's tree packing keeps samples separate. Packed log-probs match each sample run alone exactly (max |dlogp| 0.0000) when the two samples share no token. When they share a prefix (the chat-template header, or a whole GRPO prompt), the differences are bf16 rounding of the same size as two independent implementations of the same sample alone (max 0.25-0.36, mean 0.015). The negative controls move them by 2.8 to 15.4 nats. Gradients through the prefix tree match the per-sample sum to cos 0.99963.

Script: `envs/trl/packing_parity/trl_packing_parity.py`, launcher `parity.sbatch`. Results: `parity-365276.out` and `results-365276.json` (H100, hopper-dev); `parity-365268.out` is the identical earlier run without test D. TRL main @5623e8b63 unmodified, venv `bench-trl` (torch 2.14 cu130, transformers 5.20.2, fla 0.6.0).

## What the test runs

The trainer's own path, not a reimplementation:

- Model built as `AsyncGRPOTrainer.__init__` builds it: `register_tree_attention()`, `create_model_from_path(attn_implementation="tree_fa3")`, `patch_tree_gdn(model)`, `add_fused_lm_head(temperature=1.0, outputs=("log_probs", "entropy"))`, bf16 weights (the trainer's compute dtype under FSDP2 bf16 mixed precision; no FSDP on 1 GPU).
- Each row: `TreeRow.add` per sample -> `DataCollatorForRollout.torch_call` (calls `TreeRow.pack`) -> the same slicing and model call as `compute_loss` (`unpad_tree(inputs)`, `shift_labels`, `fused_lm_head=True`) -> `outputs.log_probs.flatten()[token_index]`, split per sample by `segment_id`.
- Counted in one packed forward: 18 `tree_chunk_gated_delta_rule` + 18 `tree_causal_conv1d` calls (the 18 gated-DeltaNet layers, through TRL's tree Triton kernels) and 6 `tree_attention` calls (the 6 full-attention layers, through the hub flash-attention varlen kernel).
- Samples are real: completions from TRL's own W1-S shared checkpoint-200 eval (`eval-samples-200.jsonl`), on the W1 prompt with the chat template (`enable_thinking=False`), completions cut at 6,000 tokens (3,000 in the group test), each ending in `<|im_end|>`.
- References for every sample: **alone/TRL**, the sample in a `TreeRow` of its own through the same path; **alone/HF**, a second, independent model (sdpa attention, fla `chunk_gated_delta_rule` + `causal_conv1d`, no tree kwargs, plain `lm_head` + fp32 log_softmax).
- Reported over completion tokens only.

## Results (H100, FA3 hub kernel)

| test | row | packed vs alone/TRL, max \|dlogp\| (mean) | packed vs alone/HF, max (mean) | alone/TRL vs alone/HF, max (mean) |
|---|---|---|---|---|
| D: 2 samples, raw prompts, no shared token | 12,526 tokens | **0.0000 (0.0000)**, bit-identical | | |
| A: 2 samples, 2 chat prompts | 12,548 tokens (2 shared) | 0.249 (0.0157) | 0.249 (0.0159) | 0.250 (0.0154) |
| B: 6 samples, 6 chat prompts | 37,697 tokens | 0.314 (0.0154) | 0.368 (0.0155) | 0.287 (0.0149) |
| C: prefix tree, 8 completions of 1 prompt + 1 depth-2 fork | 27,546 raw tokens folded into 26,564 | 0.362 (0.0150) | 0.300 (0.0149) | 0.304 (0.0149) |

Negative controls (same packed tokens, restarting position ids, only the boundaries removed):

| control | max \|dlogp\| vs alone | mean over the affected sample(s) |
|---|---|---|
| N1 on D: naive concatenation (tree replaced by one segment, so attention and the recurrent/conv state both cross the boundary) | **5.85** (sample 1 bit-identical: 0.0000) | 0.116 |
| N1 on A: naive concatenation | **5.68** (sample 1: 0.0000) | 0.158 |
| N2 on A: recurrent-only leak (tree attention kept, gated DeltaNet + conv told sample 2 continues from sample 1) | **2.85** | 0.126 |
| N1 on C: prefix tree laid out as one chain (each sibling reads the previous sibling's state and keys) | **15.44** (samples 2-9: 6.7-15.4) | 0.20-0.48 |

Gradients on the prefix tree (C), d(sum of completion log-probs)/d(all params), packed tree vs the sum of the 9 samples run alone:

| | rel. L2 error | cosine |
|---|---|---|
| packed tree vs alone | 0.027 | 0.99963 |
| naive concatenation vs alone | 1.79 | 0.25 |
| alone rerun vs alone (backward nondeterminism) | 0.0069 | |

## Reading the numbers

- **No leakage.** With nothing shared (D), each sample is its own root segment, chunked from its own first token exactly as when alone, and the log-probs are bit-identical. That's the same result prime-rl got (0.0000). Removing the boundary on the same row moves them by up to 5.85 nats.
- **The 0.25-0.36 in A/B/C is bf16 rounding, not leakage.** All chat prompts start with the same template tokens, so the trie shares them: each sample becomes "shared header segment + its own segment", and the gated-DeltaNet chunks (64 tokens) and attention tiles start at different offsets than when the sample runs alone. Three checks say this is numerics:
  1. Sample 1 in A shows the same 0.24 max as sample 2, though nothing precedes it but its own header. In N1, where sample 1 is laid out exactly as when alone, it's bit-identical.
  2. Packed vs alone (mean 0.0150-0.0157) is the same size as alone/TRL vs alone/HF (0.0149-0.0154), two independent implementations of a single sample.
  3. The maxima sit at ~0.25 because bf16 logits in [32, 64) are spaced 0.25 apart, so one rounding step in a logit is 0.25 in log-prob. verl's packed-vs-alone check gave 0.28-0.41 for the same reason.
- **The prefix tree is correct**, including a depth-2 fork (a sample sharing the prompt and 200 completion tokens with a sibling): every sample matches alone at the noise level, and the gradient through the tree's backward (children's state gradients summed into the parent) matches the per-sample sum to cos 0.99963. The rel. L2 error of 0.027 is about 4x the backward's own run-to-run noise, consistent with the chunk-alignment rounding above.
- The recurrent-only control (N2) shows the gated-DeltaNet boundary matters on its own: with attention correct, leaking only the linear-attention and conv state moves sample 2 by up to 2.85 nats.

Row for Finding 5: **TRL | tree packing with its own Triton GDN kernel | max diff 0.0000 with no shared prefix; 0.25-0.36 (bf16, mean 0.015) when samples share a prefix, same as alone-vs-transformers; dropping the boundary moves it by 5.7-15.4 nats.**

## B200 / FA4

The task asked for B200 with FA4. All b200 nodes were taken by bench runs, so the same script ran on an H100 (hopper-dev), where `packing._flash_attn()` picks the FA3 hub kernel. The tree-packing logic, the GDN Triton kernels and the varlen arguments are the same on both. A B200 run of the same `parity.sbatch` is queued behind the node cap (waiter 365278, output will be `packing_parity/parity-<jobid>.out`).

## miles

Not tested the same way. miles' trainer is Megatron-LM + TransformerEngine inside the 46 GB Enroot image. To run the same check you'd have to initialise Megatron's parallel state on one GPU, build the Qwen3.5-2B GPTModel through mbridge from inline args (no 2B args ship), convert/load the HF checkpoint (the torch_dist conversion took 15+ min in the smoke), construct `PackedSeqParams` (thd) by hand, and apply our FA4 window fix to TE on Blackwell. That's a separate harness with its own failure modes, not a variant of this script. The code reading (envs/miles/NOTES.md: `cu_seqlens` into both conv and recurrence) and the trainer-vs-SGLang agreement (logprob abs diff 0.012, KL 3.4-5e-4) are the evidence we have for miles.
