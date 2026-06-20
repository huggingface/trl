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
"""Step 1 of the X-Token projection-matrix pipeline.

Re-tokenizes every student-vocab token with the teacher tokenizer to build a
weighted mapping matrix W ∈ R^{V_s × V_t}.  The output is a dense top-k ``.pt``
file (dict with ``"indices"`` and ``"likelihoods"`` tensors) consumed by
``GOLDConfig.xtoken_projection_matrix_path``.

Typical usage (Llama-3.2-1B student → Qwen3-4B teacher, runtime top-4):

    python build_projection_matrix.py \\
        --student-model meta-llama/Llama-3.2-1B \\
        --teacher-model Qwen/Qwen3-4B \\
        --runtime-top-k 4 \\
        --output-dir cross_tokenizer_data

Then optionally run ``reapply_exact_map.py`` (Step 2) and
``sort_and_cut_projection_matrix.py`` (Step 3) on the output, or pass the
``--runtime-top-k`` flag here to perform the trim in-place.

See https://huggingface.co/papers/2605.21699 for the algorithm details.
"""

import argparse
import difflib
import os
import re
from collections import defaultdict

import torch
import tqdm
from transformers import AutoConfig, AutoTokenizer


# ---------------------------------------------------------------------------
# Inlined helpers (no external dependencies beyond transformers + torch)
# ---------------------------------------------------------------------------


def _canonical_token(token: str, *, enabled: bool = True) -> str:
    """Normalise space/newline prefixes so two tokenizers compare consistently."""
    if not enabled or not token:
        return token
    if token.startswith((" ", "_", "▁")):
        token = "Ġ" + token[1:]
    if token in ("Ċ", "\\n", "ĉ"):
        token = "\n"
    elif token == "Ġ\n":
        token = "Ġ\n"
    return token


def _sinkhorn_rows(A, n_iters=1):
    """Normalise rows to sum to 1 (ignores all-zero rows)."""
    for _ in range(n_iters):
        row_sums = A.sum(dim=1, keepdim=True)
        safe = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
        A = A / safe
    return A


def _clean_name(name: str) -> str:
    """Strip param-count suffixes for use in filenames."""
    cleaned = re.sub(r"-?[0-9.]+[bBmM]", "", name, flags=re.IGNORECASE)
    cleaned = cleaned.replace("-Base", "").replace("-it", "").replace("-Instruct", "").strip("-_")
    if "mini" in name.lower():
        cleaned += "_mini"
    return cleaned


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _create_weight_distribution(num_tokens):
    """Exponential-decay weights summing to 1 for multi-token mappings."""
    base = 0.9
    weights = [base * (0.1**i) for i in range(num_tokens)]
    total = sum(weights)
    return [w / total for w in weights]


def _find_similar_special_tokens(tok_a, tok_b, similarity_threshold=0.3, top_k=3):
    def _is_special(t):
        return (
            (t.startswith("<|") and t.endswith("|>"))
            or (t.startswith("<") and t.endswith(">"))
            or t
            in {
                "<eos>",
                "<bos>",
                "<pad>",
                "<unk>",
                "<s>",
                "</s>",
            }
        )

    def _sim(a, b):
        seq = difflib.SequenceMatcher(None, a, b).ratio()
        kw_a = {w for w in re.sub(r"[<>|_]", " ", a.lower()).split() if len(w) > 2}
        kw_b = {w for w in re.sub(r"[<>|_]", " ", b.lower()).split() if len(w) > 2}
        kw = len(kw_a & kw_b) / len(kw_a | kw_b) if kw_a or kw_b else 0.0
        return 0.6 * seq + 0.4 * kw

    specials_a = {i: t for t, i in tok_a.get_vocab().items() if _is_special(t)}
    specials_b = {i: t for t, i in tok_b.get_vocab().items() if _is_special(t)}
    results = []
    for ia, ta in specials_a.items():
        matches = sorted(
            ((ib, tb, _sim(ta, tb)) for ib, tb in specials_b.items() if _sim(ta, tb) >= similarity_threshold),
            key=lambda x: x[2],
            reverse=True,
        )[:top_k]
        for ib, tb, sim in matches:
            results.append(
                {"student_id": ia, "student_token": ta, "teacher_id": ib, "teacher_token": tb, "similarity": sim}
            )
    return results


def _add_multitoken_mappings(
    *,
    source_tokenizer,
    target_tokenizer,
    source_vocab_size,
    source_ignore_ids,
    target_ignore_ids,
    source_role,
    transformation_counts,
    tokens_to_cut,
    use_raw_tokens,
    use_canonicalization,
):
    target_role = "teacher" if source_role == "student" else "student"
    decoded_source = {}
    for token_id in tqdm.tqdm(range(source_vocab_size), desc=f"Decode {source_role}"):
        if token_id in source_ignore_ids:
            continue
        try:
            raw = (
                source_tokenizer.convert_ids_to_tokens([token_id])[0]
                if use_raw_tokens
                else source_tokenizer.decode([token_id])
            )
            if raw.startswith("<|") and raw.endswith("|>"):
                continue
            decoded_source[token_id] = _canonical_token(raw, enabled=use_canonicalization)
        except Exception:
            continue

    examples = []
    for src_id, src_str in tqdm.tqdm(decoded_source.items(), desc=f"Map {source_role}->{target_role}"):
        tgt_ids = target_tokenizer(src_str, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        if any(t in target_ignore_ids for t in tgt_ids):
            continue
        tgt_ids = tgt_ids[:tokens_to_cut]
        weights = _create_weight_distribution(len(tgt_ids))
        for tgt_id, w in zip(tgt_ids, weights, strict=False):
            key = (src_id, tgt_id) if source_role == "student" else (tgt_id, src_id)
            transformation_counts[key] += w
        if len(tgt_ids) >= 2:
            examples.append(
                {
                    f"{source_role}_token": src_str,
                    f"{source_role}_id": src_id,
                    f"{target_role}_tokens": [target_tokenizer.decode([t]) for t in tgt_ids],
                    f"{target_role}_ids": tgt_ids,
                    "weights": weights,
                }
            )
    return examples


def build_projection_matrix(args):
    tok_s = AutoTokenizer.from_pretrained(args.student_model)
    tok_t = AutoTokenizer.from_pretrained(args.teacher_model)

    cfg_s = AutoConfig.from_pretrained(args.student_model)
    cfg_t = AutoConfig.from_pretrained(args.teacher_model)
    name_s = args.student_model.lower()
    name_t = args.teacher_model.lower()
    vocab_s = cfg_s.text_config.vocab_size if ("gemma" in name_s or "qwen3.5" in name_s) else cfg_s.vocab_size
    vocab_t = cfg_t.text_config.vocab_size if ("gemma" in name_t or "qwen3.5" in name_t) else cfg_t.vocab_size

    print(f"Student vocab: {vocab_s}  Teacher vocab: {vocab_t}")

    ignore_s = {tok_s.convert_tokens_to_ids(t) for t in ["<|endoftext|>", "<eos>"] if t in tok_s.get_vocab()}
    ignore_t = {tok_t.convert_tokens_to_ids(t) for t in ["<|endoftext|>", "<eos>"] if t in tok_t.get_vocab()}

    counts: dict = defaultdict(float)

    _add_multitoken_mappings(
        source_tokenizer=tok_s,
        target_tokenizer=tok_t,
        source_vocab_size=vocab_s,
        source_ignore_ids=ignore_s,
        target_ignore_ids=ignore_t,
        source_role="student",
        transformation_counts=counts,
        tokens_to_cut=args.tokens_to_cut,
        use_raw_tokens=args.use_raw_tokens,
        use_canonicalization=args.use_canonicalization,
    )

    if args.enable_reverse_pass:
        _add_multitoken_mappings(
            source_tokenizer=tok_t,
            target_tokenizer=tok_s,
            source_vocab_size=vocab_t,
            source_ignore_ids=ignore_t,
            target_ignore_ids=ignore_s,
            source_role="teacher",
            transformation_counts=counts,
            tokens_to_cut=args.tokens_to_cut,
            use_raw_tokens=args.use_raw_tokens,
            use_canonicalization=args.use_canonicalization,
        )

    if args.enable_special_token_mapping:
        mappings = _find_similar_special_tokens(
            tok_s,
            tok_t,
            similarity_threshold=args.special_token_similarity_threshold,
            top_k=args.special_token_top_k or args.top_k,
        )
        for m in mappings:
            counts[(m["student_id"], m["teacher_id"])] += m["similarity"] * 0.8
        print(f"Added {len(mappings)} special-token mappings")

    print(f"Total transformation entries: {len(counts)}")

    # Build dense matrix and extract top-k.
    rows = [k[0] for k in counts]
    cols = [k[1] for k in counts]
    vals = list(counts.values())
    sparse = torch.sparse_coo_tensor(
        torch.tensor([rows, cols], dtype=torch.long),
        torch.tensor(vals, dtype=torch.float32),
        (vocab_s, vocab_t),
    ).coalesce()

    dense = sparse.to_dense()
    dense = _sinkhorn_rows(dense, n_iters=1)

    k = min(args.top_k, dense.shape[1])
    top_likelihoods, top_indices = torch.topk(dense, k=k, dim=1)

    if k < args.top_k:
        pad = args.top_k - k
        top_indices = torch.cat([top_indices, torch.full((vocab_s, pad), -1, dtype=top_indices.dtype)], dim=1)
        top_likelihoods = torch.cat([top_likelihoods, torch.zeros(vocab_s, pad, dtype=top_likelihoods.dtype)], dim=1)

    if args.enable_scale_trick:
        top_likelihoods[:, -1] = 0.2
        top_likelihoods = _sinkhorn_rows(top_likelihoods, n_iters=1)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.output_filename:
        stem = args.output_filename
    else:
        stem = f"projection_map_{_clean_name(args.student_model.split('/')[-1])}_to_{_clean_name(args.teacher_model.split('/')[-1])}_multitoken_top_{args.top_k}_double"
    if args.enable_special_token_mapping:
        stem += "_special"
    if not stem.endswith(".pt"):
        stem += ".pt"
    out_path = os.path.join(args.output_dir, stem)

    torch.save(
        {
            "indices": top_indices,
            "likelihoods": top_likelihoods,
            "student_model_id": args.student_model,
            "teacher_model_id": args.teacher_model,
            "enable_scale_trick": args.enable_scale_trick,
        },
        out_path,
    )
    print(f"Saved projection matrix → {out_path}  shape: {top_indices.shape}")
    return out_path


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--student-model", required=True)
    p.add_argument("--teacher-model", required=True)
    p.add_argument("--top-k", type=int, default=32, help="Top-k teacher tokens to keep per student token")
    p.add_argument(
        "--runtime-top-k", type=int, default=None, help="If set, also run sort_and_cut to this k after building"
    )
    p.add_argument("--tokens-to-cut", type=int, default=4, help="Max target tokens for multi-token re-encoding")
    p.add_argument("--enable-scale-trick", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--enable-reverse-pass", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--enable-special-token-mapping", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-raw-tokens", action="store_true", default=False)
    p.add_argument("--use-canonicalization", action="store_true", default=False)
    p.add_argument("--special-token-similarity-threshold", type=float, default=0.3)
    p.add_argument("--special-token-top-k", type=int, default=None)
    p.add_argument("--output-dir", default="cross_tokenizer_data")
    p.add_argument("--output-filename", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build_projection_matrix(args)
    if args.runtime_top_k is not None:
        from sort_and_cut_projection_matrix import sort_and_cut

        trimmed = out.replace(".pt", f"_top{args.runtime_top_k}.pt")
        sort_and_cut(out, trimmed, args.runtime_top_k)
        print(f"Trimmed matrix → {trimmed}")
