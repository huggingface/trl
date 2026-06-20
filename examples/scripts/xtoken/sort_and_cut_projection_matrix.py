# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Step 3 of the X-Token projection-matrix pipeline.

Sorts each row of a dense top-k projection matrix by descending weight,
trims to a new (smaller) top-k, applies Sinkhorn row normalisation, and
writes the result as a new ``.pt`` file.

Run this after ``build_projection_matrix.py`` (Step 1) and optionally
``reapply_exact_map.py`` (Step 2).

Usage:

    python sort_and_cut_projection_matrix.py \\
        --initial-projection-path cross_tokenizer_data/projection_map_...pt \\
        --top_k 4

Output is saved as ``<input>_top_<k>_sorted.pt`` unless ``--output_path``
is given.  When the input matrix was built with ``--enable-scale-trick``,
the last column is preserved in its original slot even after sorting.

See https://huggingface.co/papers/2605.21699 Section 3.2.
"""

import argparse
import os
import re

import torch
import tqdm


# ---------------------------------------------------------------------------
# Inlined helper (no external dependencies)
# ---------------------------------------------------------------------------


def _sinkhorn_one_dim(A: torch.Tensor, n_iters: int = 1) -> torch.Tensor:
    """Row-normalise A in-place (ignores rows that sum to zero)."""
    for _ in range(n_iters):
        row_sums = A.sum(dim=1, keepdim=True)
        safe = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
        A = A / safe
    return A


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def sort_and_cut(input_path: str, output_path: str, new_top_k: int, *, preserve_last: bool = False, verbose: bool = True) -> None:
    """Sort rows by descending weight, trim to ``new_top_k``, normalise, save."""
    if verbose:
        print(f"Loading: {input_path}")

    data = torch.load(input_path, map_location="cpu", weights_only=False)
    if not (isinstance(data, dict) and "indices" in data and "likelihoods" in data):
        raise ValueError("Expected dict with 'indices'/'likelihoods' tensors")

    orig_idx = data["indices"]   # [V_s, old_k]
    orig_lik = data["likelihoods"]  # [V_s, old_k]
    vocab_size, orig_k = orig_idx.shape

    if verbose:
        print(f"Original shape: {orig_idx.shape}  →  new top_k: {new_top_k}  preserve_last: {preserve_last}")

    if new_top_k > orig_k:
        print(f"Warning: new_top_k ({new_top_k}) > original ({orig_k}). Padding with -1/0.")

    effective_k = min(new_top_k, orig_k)
    new_idx = torch.full((vocab_size, new_top_k), -1, dtype=orig_idx.dtype)
    new_lik = torch.zeros(vocab_size, new_top_k, dtype=orig_lik.dtype)

    for row in tqdm.tqdm(range(vocab_size), desc="Sort & cut", disable=not verbose):
        r_idx = orig_idx[row]
        r_lik = orig_lik[row]

        valid = (r_idx != -1) & (r_lik > 0)
        if not valid.any():
            continue

        v_idx = r_idx[valid]
        v_lik = r_lik[valid]

        if preserve_last and new_top_k >= 2:
            # Keep the last original column anchored at position new_top_k-1.
            # Sort only the first (orig_k-1) valid elements, take top (new_top_k-1) of them.
            last_i = orig_idx[row, orig_k - 1]
            last_l = orig_lik[row, orig_k - 1]

            # valid elements excluding the last original column
            sort_valid = valid.clone()
            sort_valid[orig_k - 1] = False
            sv_idx = r_idx[sort_valid]
            sv_lik = r_lik[sort_valid]

            if sv_lik.numel() > 0:
                sorted_l, order = torch.sort(sv_lik, descending=True)
                sorted_i = sv_idx[order]
                n = min(len(sorted_i), new_top_k - 1)
                new_idx[row, :n] = sorted_i[:n]
                new_lik[row, :n] = sorted_l[:n]

            if last_i != -1 and last_l > 0:
                new_idx[row, new_top_k - 1] = last_i
                new_lik[row, new_top_k - 1] = last_l
        else:
            sorted_l, order = torch.sort(v_lik, descending=True)
            sorted_i = v_idx[order]
            n = min(len(sorted_i), effective_k)
            new_idx[row, :n] = sorted_i[:n]
            new_lik[row, :n] = sorted_l[:n]

    norm_lik = _sinkhorn_one_dim(new_lik.clone(), n_iters=1)

    out_data = {"indices": new_idx, "likelihoods": norm_lik}
    for k, v in data.items():
        if k not in ("indices", "likelihoods"):
            out_data[k] = v

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(out_data, output_path)
    if verbose:
        print(f"Saved → {output_path}  shape: {new_idx.shape}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_output_path(input_path: str, new_top_k: int, preserve_last: bool) -> str:
    d = os.path.dirname(input_path)
    base, ext = os.path.splitext(os.path.basename(input_path))
    base = re.sub(r"_top_\d+", "", base)
    suffix = "_sorted" + ("_preservelast" if preserve_last else "")
    return os.path.join(d, f"{base}_top_{new_top_k}{suffix}{ext or '.pt'}")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--initial-projection-path", required=True)
    p.add_argument("--top_k", type=int, required=True, help="New top-k cutoff")
    p.add_argument("--output_path", default=None)
    p.add_argument(
        "--preserve_last",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force enable/disable preserve-last. Defaults to reading enable_scale_trick from the input file.",
    )
    p.add_argument("--quiet", "-q", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve preserve_last: CLI flag > metadata > False.
    if args.preserve_last is None:
        try:
            meta = torch.load(args.initial_projection_path, map_location="cpu", weights_only=False)
            preserve_last = bool(meta.get("enable_scale_trick", False)) if isinstance(meta, dict) else False
        except Exception:
            preserve_last = False
        if preserve_last and not args.quiet:
            print("Auto-enabling --preserve_last (input has enable_scale_trick=True)")
    else:
        preserve_last = args.preserve_last

    output_path = args.output_path or _resolve_output_path(args.initial_projection_path, args.top_k, preserve_last)
    sort_and_cut(args.initial_projection_path, output_path, args.top_k, preserve_last=preserve_last, verbose=not args.quiet)
    if not args.quiet:
        print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
