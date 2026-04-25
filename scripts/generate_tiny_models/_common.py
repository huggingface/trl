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

# Shared utilities for the tiny-model generation scripts in this directory.
# Each sibling script builds a single tiny model and pushes it to the Hub under
# the `trl-internal-testing` organization.

import argparse
import json
import os
import struct
import tempfile

import torch
from huggingface_hub import CommitOperationAdd, HfApi, ModelCard
from packaging.version import Version
from torch import nn
from transformers import AutoConfig, ProcessorMixin


ORGANIZATION = "trl-internal-testing"

MODEL_CARD = """
---
library_name: transformers
tags: [trl]
---

# Tiny {model_class_name}

This is a minimal model built for unit tests in the [TRL](https://github.com/huggingface/trl) library.
"""


api = HfApi()


def check_transformers_version(expected_version):
    """Raise unless the installed transformers matches `expected_version` exactly."""
    import transformers

    if Version(transformers.__version__) != Version(expected_version):
        raise RuntimeError(
            f"This script requires transformers=={expected_version}, but {transformers.__version__} is installed."
        )


def smoke_test(model, tokenizer_or_processor=None):
    """Run a minimal forward pass to sanity-check the tiny model doesn't crash or produce NaNs."""
    model.eval()
    device = next(model.parameters()).device

    if isinstance(tokenizer_or_processor, ProcessorMixin):
        # VLM path: build a dummy (image, text) input via the processor.
        from PIL import Image

        processor = tokenizer_or_processor
        red = Image.new("RGB", (24, 24), color="red")
        blue = Image.new("RGB", (24, 24), color="blue")
        messages = [
            [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": red}, {"type": "text", "text": "What is this?"}],
                }
            ],
            [{"role": "user", "content": [{"type": "text", "text": "Is it blue?"}, {"type": "image", "image": blue}]}],
        ]
        inputs = processor.apply_chat_template(
            conversation=messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(device)
    else:
        inputs = {"input_ids": torch.tensor([[1, 2, 3, 4]], device=device)}

    with torch.no_grad():
        out = model(**inputs)

    if "logits" in out:
        output_tensor = out["logits"]
    elif "last_hidden_state" in out:
        output_tensor = out["last_hidden_state"]
    else:
        raise RuntimeError(f"[smoke_test] {model.__class__.__name__}: no logits or last_hidden_state on output")
    if torch.isnan(output_tensor).any():
        raise RuntimeError(f"[smoke_test] {model.__class__.__name__}: NaN in forward output")
    print(f"[smoke_test] {model.__class__.__name__}: OK (output shape {tuple(output_tensor.shape)})")


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


_DIFF_IGNORE = {"_name_or_path", "transformers_version", "architectures", "model_type", "torch_dtype", "dtype"}


def _read_local_safetensors_dtypes(path):
    """Parse the header of every .safetensors file under `path` and return {tensor_name: dtype_str}."""
    dtypes = {}
    for fname in os.listdir(path):
        if not fname.endswith(".safetensors"):
            continue
        with open(os.path.join(path, fname), "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            dtypes[name] = info["dtype"]
    return dtypes


def check_dtype_pattern(reference_id, model, revision=None):
    """Flag tensors whose dtype diverges from the reference checkpoint.

    Saves the tiny to a temp dir first so its on-disk tensor names (after any save-time conversion
    applied by the model class) line up with the reference's safetensors names. When the installed
    transformers exposes ``conversion_mapping`` (>= 5.0.0), also index reference tensors under their
    forward-mapped (internal) names so models whose save path doesn't reverse every renaming (e.g.
    V4's complex multi-group rules) still match. Useful to catch cases like Qwen3.5 where specific
    params are kept in fp32 while the rest is bf16.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)
        tiny_dtypes = _read_local_safetensors_dtypes(tmpdir)

    ref_metadata = api.get_safetensors_metadata(reference_id, revision=revision)
    ref_dtypes = {name: info.dtype for fm in ref_metadata.files_metadata.values() for name, info in fm.tensors.items()}

    try:
        from transformers.conversion_mapping import get_checkpoint_conversion_mapping
        from transformers.core_model_loading import WeightRenaming

        renamings = [r for r in (get_checkpoint_conversion_mapping(model.config.model_type) or []) if isinstance(r, WeightRenaming)]
    except ImportError:
        renamings = []
    for name, dtype in list(ref_dtypes.items()):
        renamed = name
        for r in renamings:
            renamed, _ = r.rename_source_key(renamed)
        if renamed != name:
            ref_dtypes.setdefault(renamed, dtype)

    mismatches = []
    unmatched = []
    for name, tiny_dtype in tiny_dtypes.items():
        ref_dtype = ref_dtypes.get(name)
        if ref_dtype is None:
            unmatched.append(name)  # tensor has no counterpart in the reference
            continue
        if tiny_dtype != ref_dtype:
            mismatches.append((name, ref_dtype, tiny_dtype))

    total = len(tiny_dtypes)
    matched = total - len(unmatched)
    print(f"[dtype_check] {reference_id}: {matched}/{total} tiny tensors matched reference")
    if unmatched:
        print(f"[dtype_check]   {len(unmatched)} tiny tensors have no reference match:")
        for name in unmatched:
            print(f"    {name}")
    if mismatches:
        print(f"[dtype_check]   {len(mismatches)} tensors differ from reference:")
        for name, ref, tiny in mismatches:
            print(f"    {name}: reference={ref}, tiny={tiny}")
    else:
        print(f"[dtype_check]   all {matched} matched tensors have the reference dtype")


def print_config_diff(reference_id, model, revision=None):
    """Print the flat, recursive diff between the reference Hub config and the tiny-model config."""
    reference_config = AutoConfig.from_pretrained(reference_id, revision=revision)
    ref_flat = _flatten(reference_config.to_dict())
    tiny_flat = _flatten(model.config.to_dict())

    keys = sorted(set(ref_flat) | set(tiny_flat))
    rows = []
    for k in keys:
        if any(k == ig or k.endswith(f".{ig}") for ig in _DIFF_IGNORE):
            continue
        rv, tv = ref_flat.get(k, "<missing>"), tiny_flat.get(k, "<missing>")
        if rv != tv:
            rows.append((k, rv, tv))

    print(f"[config_diff] {reference_id} vs tiny ({len(rows)} differences)")
    for k, r, t in rows:
        print(f"  {k:48s} {str(r)[:34]:34s} → {str(t)[:34]}")


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="If the repo already exists, open a PR instead of skipping.",
    )
    args, _ = parser.parse_known_args()
    return args


def push_to_hub(model, tokenizer, generation_config, prefix=None, suffix=None, create_pr=None):
    if create_pr is None:
        create_pr = _parse_args().create_pr

    model_class_name = model.__class__.__name__
    content = MODEL_CARD.format(model_class_name=model_class_name)
    model_card = ModelCard(content)
    if prefix is not None:
        model_class_name = f"{prefix}-{model_class_name}"
    repo_id = f"{ORGANIZATION}/{model_class_name}"
    if suffix is not None:
        repo_id += f"-{suffix}"

    exists = api.repo_exists(repo_id)
    if exists and not create_pr:
        print(f"Model {repo_id} already exists, skipping (pass --create-pr to open a PR)")
        return

    if not exists:
        api.create_repo(repo_id, exist_ok=True)

    # Save all artifacts to a temp dir and upload them in a single commit, so --create-pr opens one PR.
    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)
        if tokenizer is not None:
            tokenizer.save_pretrained(tmpdir)
        if generation_config is not None:
            generation_config.save_pretrained(tmpdir)
        model_card.save(os.path.join(tmpdir, "README.md"))

        operations = [
            CommitOperationAdd(
                path_in_repo=os.path.relpath(os.path.join(root, name), tmpdir),
                path_or_fileobj=os.path.join(root, name),
            )
            for root, _, files in os.walk(tmpdir)
            for name in files
        ]
        commit_info = api.create_commit(
            repo_id=repo_id,
            operations=operations,
            commit_message=f"Upload {model.__class__.__name__}",
            create_pr=exists and create_pr,
        )
        if commit_info.pr_url:
            print(f"[push_to_hub] PR opened: {commit_info.pr_url}")


def init_weights_tiny_model(model):
    """
    Initialize tiny test models to avoid NaNs from uninitialized weights.

    Uses safe defaults:
      - Linear/Conv1d: Xavier uniform (weights), zero (biases)
      - Embedding: Normal(0, 0.02)
      - LayerNorm: Ones (weights), zero (biases)
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            nn.init.xavier_uniform_(module.weight)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Conv1d):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            nn.init.xavier_uniform_(module.weight)
