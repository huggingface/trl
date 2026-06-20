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
"""Step 2 (optional) of the X-Token projection-matrix pipeline.

Overwrites exact-match rows of a dense top-k projection matrix so that each student token that maps exactly to a
teacher token is encoded as a one-hot assignment (likelihood 1.0 at the matching teacher id, sentinel -1 and weight 0.0
everywhere else).

Running this between Steps 1 and 3 improves H-KL results by ensuring exact matches are never lost to Sinkhorn
normalisation or the top-k trim.

Usage:

    python reapply_exact_map.py \\
        --student-model meta-llama/Llama-3.2-1B \\ --teacher-model Qwen/Qwen3-4B \\ --initial-projection-path
        cross_tokenizer_data/projection_map_...pt

Output is saved as ``<input>_exact_map_remapped.pt``.

See https://huggingface.co/papers/2605.21699 Section 3.2.
"""

import argparse
import os

import torch
from transformers import AutoTokenizer


def _canonical_token(token: str, *, enabled: bool = True) -> str:
    if not enabled or not token:
        return token
    if token.startswith((" ", "_", "▁")):
        token = "Ġ" + token[1:]
    if token in ("Ċ", "\\n", "ĉ"):
        token = "\n"
    return token


def reapply_exact_map(args):
    tok_s = AutoTokenizer.from_pretrained(args.student_model)
    tok_t = AutoTokenizer.from_pretrained(args.teacher_model)

    tokens_s = [
        _canonical_token(tok_s.convert_ids_to_tokens([i])[0], enabled=args.use_canonicalization)
        for i in range(len(tok_s))
    ]
    tokens_t = [
        _canonical_token(tok_t.convert_ids_to_tokens([j])[0], enabled=args.use_canonicalization)
        for j in range(len(tok_t))
    ]
    teacher_map = {t: j for j, t in enumerate(tokens_t)}

    match_s, match_t = [], []
    for i, ts in enumerate(tokens_s):
        if ts in teacher_map:
            match_s.append(i)
            match_t.append(teacher_map[ts])

    print(f"Found {len(match_s)} exact-match token pairs")

    data = torch.load(args.initial_projection_path, map_location="cpu", weights_only=False)
    if not (isinstance(data, dict) and "indices" in data and "likelihoods" in data):
        raise ValueError(f"Expected dict with 'indices'/'likelihoods' tensors, got {type(data).__name__}")

    for s_id, t_id in zip(match_s, match_t, strict=False):
        remapped_idx = torch.full_like(data["indices"][s_id], -1)
        remapped_lik = torch.zeros_like(data["likelihoods"][s_id])
        remapped_idx[0] = t_id
        remapped_lik[0] = 1.0
        data["indices"][s_id] = remapped_idx
        data["likelihoods"][s_id] = remapped_lik

    base, ext = os.path.splitext(args.initial_projection_path)
    out_path = base + "_exact_map_remapped" + (ext or ".pt")
    torch.save(data, out_path)
    print(f"Saved remapped projection matrix → {out_path}")
    return out_path


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--student-model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--teacher-model", default="Qwen/Qwen3-4B")
    p.add_argument("--initial-projection-path", required=True)
    p.add_argument("--use-canonicalization", action="store_true", default=False)
    return p.parse_args()


if __name__ == "__main__":
    reapply_exact_map(parse_args())
