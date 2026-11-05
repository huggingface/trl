# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Common-support targets for HTTP teacher-forced distillation.

Table preparation adapted from NVIDIA-NeMo/RL PR #4680 (547d391),
``tools/csd/build_subtoks_table.py``. Tables are built once per tokenizer pair
in the CPU rollout worker; no NeMo runtime dependency is required.
"""

import math
from collections import defaultdict

import numpy as np
import torch
from tokenizers.decoders import ByteLevel
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerFast


def _expm1_minus_x(x):
    # Removing the linear term makes the divergence and its gradient vanish at
    # equality. The series also avoids cancellation in expm1(x) - x near zero.
    return torch.where(
        x.abs() < 1e-4,
        x.square() * (0.5 + x / 6 + x.square() / 24),
        x.expm1() - x,
    )


def csd_divergence(student_logps, teacher_logps, mask, beta):
    """Per-chunk divergence resistant to softmax cancellation.

    Implements the paired log-ratio idea from Barron et al.
    (https://jonbarron.info/softmaxfix/#sec:bug2) on CSD events plus REST.
    Normalized KL is evaluated as an f-divergence with its linear term removed,
    so equal event distributions have exactly zero loss and gradient. Only the
    small event support uses fp64; vocabulary projections remain fp32. This
    cannot recover precision already lost in the HTTP teacher's log-probabilities.

    Args:
        student_logps (`torch.Tensor`):
            Student event log-probabilities, without REST, of shape `(chunks, support)`.
        teacher_logps (`torch.Tensor`):
            Detached teacher log-probabilities of the corresponding events.
        mask (`torch.Tensor`):
            Boolean mask of real events; padding contributes neither mass nor gradient.
        beta (`float`):
            Forward KL at `0`, reverse KL at `1`, otherwise generalized JSD.

    Returns:
        `tuple` of per-chunk divergence and detached teacher entropy over events plus REST.
    """
    s = student_logps.double().masked_fill(~mask, -torch.inf)
    t = teacher_logps.detach().double().masked_fill(~mask, -torch.inf)
    # Avoid subtracting a rounded support mass from one. A tiny fp64 floor
    # covers rounding at complete support, rather than imposing a 1e-7 tail.
    s_rest = torch.log(-torch.expm1(torch.logsumexp(s, -1, keepdim=True).clamp(max=-torch.finfo(s.dtype).eps)))
    t_rest = torch.log(-torch.expm1(torch.logsumexp(t, -1, keepdim=True).clamp(max=-torch.finfo(t.dtype).eps)))
    valid = torch.cat([mask, torch.ones_like(mask[:, :1])], -1)
    s = torch.cat([s, s_rest], -1).masked_fill(~valid, 0.0)
    t = torch.cat([t, t_rest], -1).masked_fill(~valid, 0.0)
    q = torch.where(valid, s.exp(), 0.0)
    p = torch.where(valid, t.exp(), 0.0)
    delta = t - s
    if beta == 0:
        # KL(p || q) = sum p * (expm1(log(q/p)) - log(q/p)).
        terms = torch.where(delta.abs() < 1, p * _expm1_minus_x((-delta).clamp(-1, 1)), p * delta + q - p)
    elif beta == 1:
        terms = torch.where(delta.abs() < 1, q * _expm1_minus_x(delta.clamp(-1, 1)), -q * delta + p - q)
    else:
        # log(m/q) and log(m/p) from paired ratios, avoiding subtraction of
        # independently rounded probabilities or mixture log-probabilities.
        log_m_q = torch.where(
            delta.abs() < 1,
            torch.log1p(beta * delta.clamp(-1, 1).expm1()),
            torch.logaddexp(torch.full_like(delta, math.log1p(-beta)), delta + math.log(beta)),
        )
        log_m_p = torch.where(
            delta.abs() < 1,
            torch.log1p((1 - beta) * (-delta).clamp(-1, 1).expm1()),
            torch.logaddexp(torch.full_like(delta, math.log(beta)), -delta + math.log1p(-beta)),
        )
        mixture = (1 - beta) * q + beta * p
        kl_q = torch.where(log_m_q.abs() < 1, q * _expm1_minus_x(log_m_q.clamp(-1, 1)), -q * log_m_q + mixture - q)
        kl_p = torch.where(log_m_p.abs() < 1, p * _expm1_minus_x(log_m_p.clamp(-1, 1)), -p * log_m_p + mixture - p)
        terms = (1 - beta) * kl_q + beta * kl_p
    loss = terms.masked_fill(~valid, 0.0).sum(-1).to(student_logps.dtype)
    entropy = -(p * t).masked_fill(~valid, 0.0).sum(-1).to(student_logps.dtype)
    return loss, entropy


def special_roles(tokenizer: PreTrainedTokenizerFast) -> dict[str, int | None]:
    """Return semantic roles independently of their token spellings."""
    return {
        "bos": tokenizer.bos_token_id,
        "eos": tokenizer.eos_token_id,
        "pad": tokenizer.pad_token_id,
        "unk": tokenizer.unk_token_id,
        "sep": tokenizer.sep_token_id,
        "cls": tokenizer.cls_token_id,
        "mask": tokenizer.mask_token_id,
    }


def hard_special_ids(tokenizer: PreTrainedTokenizerFast) -> set[int]:
    """Include hard added specials without excluding ordinary added tokens."""
    return set(tokenizer.all_special_ids) | {
        token_id for token_id, token in tokenizer.added_tokens_decoder.items() if token.special
    }


def byte_bpe_model(tokenizer: PreTrainedTokenizerFast) -> BPE | None:
    """Return the byte-preserving BPE model, when this tokenizer has one."""
    backend = tokenizer.backend_tokenizer
    if isinstance(backend.model, BPE) and isinstance(backend.decoder, ByteLevel):
        return backend.model
    return None


def require_supported_tokenizer(tokenizer: PreTrainedTokenizerFast, *, label: str) -> None:
    """Reject decoders whose single-token text can change with context.

    Metaspace and WordPiece decoders can add or remove whitespace or word
    markers depending on token position. Isolated decoding cannot establish
    the complete events needed by a static table for those tokenizers.
    """
    if byte_bpe_model(tokenizer) is None:
        raise ValueError(
            f"{label}: CSD table preparation and validation require a fast "
            "tokenizer with a BPE model and ByteLevel decoder. Other decoder "
            "families, including Metaspace and WordPiece, are unsupported "
            "because isolated token decoding can change events with context."
        )


def role_mapping(source: PreTrainedTokenizerFast, destination: PreTrainedTokenizerFast) -> dict[int, int]:
    """Map unambiguous shared special roles; never resolve aliases by order."""
    destination_roles = special_roles(destination)
    candidates: dict[int, set[int]] = {}
    for role, source_id in special_roles(source).items():
        destination_id = destination_roles[role]
        if source_id is not None and destination_id is not None:
            candidates.setdefault(source_id, set()).add(destination_id)
    return {source_id: next(iter(targets)) for source_id, targets in candidates.items() if len(targets) == 1}


def special_match_is_valid(
    source: PreTrainedTokenizerFast,
    destination: PreTrainedTokenizerFast,
    *,
    source_id: int,
    destination_id: int,
) -> bool:
    """Permit identical hard-special spellings only when roles do not conflict."""
    source_roles = {k for k, v in special_roles(source).items() if v == source_id}
    destination_roles = {k for k, v in special_roles(destination).items() if v == destination_id}
    return not (
        source_roles and destination_roles and source_roles.isdisjoint(destination_roles)
    ) and source.convert_ids_to_tokens(source_id) == destination.convert_ids_to_tokens(destination_id)


def ordinary_event_matches(
    source: PreTrainedTokenizerFast,
    destination: PreTrainedTokenizerFast,
    *,
    source_id: int,
    destination_ids: list[int],
) -> bool:
    """Check complete events without replacing incomplete UTF-8 fragments.

    Byte-level BPE pairs compare their reversible byte surfaces. Ordinary added
    tokens compare decoded text and conservatively reject replacement characters.
    """
    if not destination_ids:
        return False
    if (
        byte_bpe_model(source) is not None
        and byte_bpe_model(destination) is not None
        and source_id not in source.added_tokens_decoder
        and not any(i in destination.added_tokens_decoder for i in destination_ids)
    ):
        return source.convert_ids_to_tokens(source_id) == "".join(destination.convert_ids_to_tokens(destination_ids))
    source_text = source.decode([source_id], clean_up_tokenization_spaces=False)
    destination_text = destination.decode(destination_ids, clean_up_tokenization_spaces=False)
    return bool(source_text) and "\ufffd" not in source_text and source_text == destination_text


def build_subtoks_table(
    source: PreTrainedTokenizerFast,
    destination: PreTrainedTokenizerFast,
    *,
    max_subtokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build complete destination chains for every source vocabulary ID.

    Args:
        source: Source tokenizer, including its added tokens.
        destination: Tokenizer whose IDs populate each chain.
        max_subtokens: Positive maximum chain length; longer events are unmapped.

    Returns:
        Integer tensors ``subtoks[V_source, max_subtokens]`` and
        ``lengths[V_source]``, ready for the CSD loss table loader.
    """
    if max_subtokens < 1:
        raise ValueError("max_subtokens must be positive")
    require_supported_tokenizer(source, label="source tokenizer")
    require_supported_tokenizer(destination, label="destination tokenizer")
    subtoks = torch.full((len(source), max_subtokens), -1, dtype=torch.long)
    lengths = torch.zeros(len(source), dtype=torch.long)
    source_specials = hard_special_ids(source)
    destination_specials = hard_special_ids(destination)
    roles = role_mapping(source, destination)
    destination_vocab = destination.get_vocab()
    destination_bpe = byte_bpe_model(destination)
    byte_pair = byte_bpe_model(source) is not None and destination_bpe is not None

    for source_id in range(len(source)):
        surface = source.convert_ids_to_tokens(source_id)
        exact_id = destination_vocab.get(surface)
        destination_ids: list[int] = []
        if source_id in source_specials:
            if source_id in roles:
                destination_ids = [roles[source_id]]
            elif (
                exact_id is not None
                and exact_id in destination_specials
                and special_match_is_valid(source, destination, source_id=source_id, destination_id=exact_id)
            ):
                destination_ids = [exact_id]
        else:
            if (
                exact_id is not None
                and exact_id not in destination_specials
                and ordinary_event_matches(source, destination, source_id=source_id, destination_ids=[exact_id])
            ):
                destination_ids = [exact_id]
            else:
                text = source.decode([source_id], clean_up_tokenization_spaces=False)
                candidate = destination.encode(text, add_special_tokens=False)
                if (
                    byte_pair
                    and source_id not in source.added_tokens_decoder
                    and not ordinary_event_matches(
                        source,
                        destination,
                        source_id=source_id,
                        destination_ids=candidate,
                    )
                ):
                    # Token-level Unicode decoding can replace incomplete byte
                    # fragments. BPE on the reversible surface preserves them.
                    assert destination_bpe is not None
                    candidate = [piece.id for piece in destination_bpe.tokenize(surface)]
                if not any(i in destination_specials for i in candidate) and ordinary_event_matches(
                    source, destination, source_id=source_id, destination_ids=candidate
                ):
                    destination_ids = candidate
        if 0 < len(destination_ids) <= max_subtokens:
            lengths[source_id] = len(destination_ids)
            subtoks[source_id, : len(destination_ids)] = torch.tensor(destination_ids)
    return subtoks, lengths


class CSDTargetBuilder:
    """Prepare sparse common-support events without changing generated student token IDs.

    Args:
        student_tokenizer (`PreTrainedTokenizerFast`):
            Tokenizer used by the student generator and trainer.
        teacher_tokenizer (`PreTrainedTokenizerFast`):
            Tokenizer used by the routed teacher server.
        top_k (`int`):
            Maximum event support, including the realized pair and excluding REST.
    """

    def __init__(self, student_tokenizer, teacher_tokenizer, top_k):
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.top_k = top_k
        self.forward, self.forward_lengths = build_subtoks_table(student_tokenizer, teacher_tokenizer, max_subtokens=8)
        self.reverse, self.reverse_lengths = build_subtoks_table(teacher_tokenizer, student_tokenizer, max_subtokens=8)
        self.common = {}
        self.prefix = defaultdict(list)
        for sid in (self.forward_lengths > 0).nonzero().flatten().tolist():
            chain = self.forward[sid, : self.forward_lengths[sid]].tolist()
            if len(chain) == 1:
                self.common[chain[0]] = sid
            else:
                self.prefix[tuple(chain[:-1])].append((sid, chain[-1]))

        # ByteLevel's alphabet is reversible even for tokens that contain only
        # part of a UTF-8 character. Added tokens instead contain literal text.
        byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        remaining = [i for i in range(256) if i not in byte_values]
        alphabet = byte_values + remaining
        characters = byte_values + list(range(256, 256 + len(remaining)))
        self.byte_decoder = dict(zip(map(chr, characters), alphabet, strict=True))
        self.student_surfaces = self._surfaces(student_tokenizer)
        self.teacher_surfaces = self._surfaces(teacher_tokenizer)

    def _surfaces(self, tokenizer):
        added = tokenizer.added_tokens_decoder
        return [
            added[i].content.encode("utf-8")
            if i in added
            else bytes(self.byte_decoder[c] for c in tokenizer.convert_ids_to_tokens(i))
            for i in range(len(tokenizer))
        ]

    def align(self, student_ids, teacher_ids):
        """Return half-open token spans covering identical byte strings.

        The inputs contain assistant content only. Terminal special tokens are
        paired separately by the caller. A non-round-tripping completion has no
        valid chunks; it must never acquire targets for different text.
        """
        s_parts = [self.student_surfaces[i] for i in student_ids]
        t_parts = [self.teacher_surfaces[i] for i in teacher_ids]
        if b"".join(s_parts) != b"".join(t_parts):
            return []
        s_ends = np.cumsum([len(part) for part in s_parts])
        t_ends = np.cumsum([len(part) for part in t_parts])
        s_start = t_start = s = t = 0
        chunks = []
        while s < len(s_ends) and t < len(t_ends):
            if s_ends[s] < t_ends[t]:
                s += 1
            elif s_ends[s] > t_ends[t]:
                t += 1
            else:
                chunks.append((s_start, s + 1, t_start, t + 1))
                s += 1
                t += 1
                s_start, t_start = s, t
        return chunks

    def build(self, student_ids, teacher_ids, chunks, topk_ids, topk_logprobs, mismatch_weight):
        """Map teacher rows to student events and fold in observed teacher prefixes.

        Args:
            student_ids (`list[int]`):
                Original student completion token IDs.
            teacher_ids (`list[int]`):
                Teacher completion token IDs for the same content.
            chunks (`list[tuple[int, int, int, int]]`):
                Half-open student and teacher spans, relative to their completions.
            topk_ids (`list[list[int]]`):
                Teacher candidates at each teacher completion label position.
            topk_logprobs (`list[list[float]]`):
                Full-vocabulary-normalized teacher log-probabilities at those positions.
            mismatch_weight (`float`):
                Relative loss weight of a chunk spanning multiple tokens on either side.

        Returns:
            `tuple` containing student candidate IDs, teacher event log-probabilities,
            student prefix lengths, and chunk weights, all on the student token axis.
        """
        candidates = np.full((len(student_ids), self.top_k + 1), -1, dtype=np.int64)
        logprobs = np.full(candidates.shape, -np.inf, dtype=np.float32)
        prefix_lengths = np.zeros(len(student_ids), dtype=np.int64)
        weights = np.zeros(len(student_ids), dtype=np.float32)
        rows = [dict(zip(ids, lps, strict=True)) for ids, lps in zip(topk_ids, topk_logprobs, strict=True)]
        for s0, s1, t0, t1 in chunks:
            s_chain, t_chain = student_ids[s0:s1], teacher_ids[t0:t1]
            if any(tid not in rows[pos] for pos, tid in enumerate(t_chain, start=t0)):
                continue
            teacher_prefix = sum(rows[pos][tid] for pos, tid in enumerate(t_chain[:-1], start=t0))
            possible = []
            if len(s_chain) == len(t_chain) == 1:
                possible = [(self.common[tid], tid) for tid in rows[t1 - 1] if tid in self.common]
            elif len(s_chain) == 1:
                possible = self.prefix[tuple(t_chain[:-1])]
            elif len(t_chain) == 1:
                for tid in rows[t1 - 1]:
                    length = int(self.reverse_lengths[tid])
                    chain = self.reverse[tid, :length].tolist()
                    if length == len(s_chain) and chain[:-1] == s_chain[:-1]:
                        possible.append((chain[-1], tid))
            # Prefix alternatives use exactly the observed prefix. No model
            # forward on a hypothetical continuation is needed on either side.
            alternatives = [
                (sid, tid)
                for sid, tid in possible
                if sid != s_chain[-1] and tid != t_chain[-1] and tid in rows[t1 - 1]
            ]
            alternatives.sort(key=lambda pair: rows[t1 - 1][pair[1]], reverse=True)
            pairs = [(s_chain[-1], t_chain[-1])]
            seen_s, seen_t = {s_chain[-1]}, {t_chain[-1]}
            for sid, tid in alternatives:
                if len(pairs) >= self.top_k:
                    break
                if sid not in seen_s and tid not in seen_t:
                    pairs.append((sid, tid))
                    seen_s.add(sid)
                    seen_t.add(tid)
            for k, (sid, tid) in enumerate(pairs):
                candidates[s1 - 1, k] = sid
                logprobs[s1 - 1, k] = teacher_prefix + rows[t1 - 1][tid]
            prefix_lengths[s1 - 1] = len(s_chain) - 1
            weights[s1 - 1] = 1.0 if len(s_chain) == len(t_chain) == 1 else mismatch_weight
        return candidates, logprobs, prefix_lengths, weights
