#!/usr/bin/env python3
"""
Async CAFA‑5 inference against a single vLLM server
- Builds batched /generate requests with protein sequences
- Streams to http://HOST:PORT/generate/
"""

import argparse, asyncio, time, json, aiohttp
from typing import List, Dict, Any
import os
import numpy as np
from bioreason2.dataset.cafa5.collate import _coords_from_cif, _coords_from_pdb
from bioreason2.dataset.cafa5.load import load_cafa5_dataset
from bioreason2.utils import str2bool
 

def add_structures_to_dataset(dataset, max_length_protein: int = 2048, num_proc: int = 192):
    """
    Add structure coordinates to dataset using the same logic as collate function.
    """
    def process_structure(example):
        struct_path = example.get("structure_path")
        
        # Same logic as collate function lines 122-142
        if struct_path is not None and os.path.exists(struct_path):
            try:
                if struct_path.endswith(".cif"):
                    coords = _coords_from_cif(struct_path)
                elif struct_path.endswith(".pdb"):
                    coords = _coords_from_pdb(struct_path)
                else:
                    raise ValueError(f"Unsupported structure format: {struct_path}")
            except Exception:
                # On error, fall back to empty coordinates
                coords = np.full((0, 3, 3), np.nan)
        else:
            coords = np.full((0, 3, 3), np.nan)
        
        # Truncate if number of residues exceeds max_length_protein
        if coords.shape[0] > max_length_protein:
            coords = coords[:max_length_protein]
        
        # For empty coordinates, return None (helps schema alignment across shards)
        example["structure_coords"] = None if coords.shape[0] == 0 else coords.tolist()
        
        # Fix sequence field name consistency
        if "sequence" in example and "protein_sequences" not in example:
            example["protein_sequences"] = [example["sequence"]]
            
        return example
    
    print(f"Adding structure coordinates to {len(dataset)} samples using {num_proc} processes...")
    print(f"Expected processing time: ~{len(dataset) / (num_proc * 80):.1f} minutes (estimated)")
    return dataset.map(
        process_structure,
        num_proc=num_proc,
        desc="Adding structures"
    )


try:
    import orjson  # type: ignore
    _dumps = lambda o: orjson.dumps(o).decode()
except Exception:
    _dumps = lambda o: json.dumps(o)


async def _post(session: aiohttp.ClientSession, url: str, payload: Dict[str, Any], timeout_s: int):
    async with session.post(url, json=payload, timeout=timeout_s) as resp:
        resp.raise_for_status()
        return await resp.json()


def _flatten_user_messages_to_text(prompt) -> str:
    def _to_text(item):
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            if item.get("type") == "text":
                return item.get("text", "")
            if "content" in item:
                return _to_text(item["content"])
        if isinstance(item, list):
            return " ".join(_to_text(x) for x in item)
        return ""

    if isinstance(prompt, list):
        user_msgs = [m for m in prompt if m.get("role") == "user"]
        user_texts = []
        for m in user_msgs:
            user_texts.append(_to_text(m.get("content", "")))
        return "\n".join(t.strip() for t in user_texts if t and t.strip())
    return str(prompt)


def build_batches(samples, batch_size: int, temperature: float, top_p: float,
                  max_new_tokens: int, repetition_penalty: float):
    batches = []
    for i in range(0, len(samples), batch_size):
        end = min(i + batch_size, len(samples))
        batch_ds = samples.select(range(i, end))
        prompts, protein_seqs, go_aspects, structure_coords = [], [], [], []

        for s in batch_ds:
            prompts.append(_flatten_user_messages_to_text(s["prompt"]))
            protein_seqs.append(s.get("protein_sequences", []))
            go_aspects.append(s.get("go_aspect") or None)
            structure_coords.append(s.get("structure_coords") or None)

        payload = {
            "prompts": prompts,
            "protein_sequences": protein_seqs,
            "go_aspects": go_aspects if any(a is not None for a in go_aspects) else None,
            "structure_coords": structure_coords if any(c is not None for c in structure_coords) else None,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            "generation_kwargs": {},
        }
        batches.append(payload)

    return batches


async def main(args):
    server = f"http://{args.host}:{args.port}"
    print(f"🚀 Server → {server}")

    print("📥 Loading CAFA‑5 validation split …")
    _, val_ds, _ = load_cafa5_dataset(
        dataset=args.cafa5_dataset,
        dataset_name=args.cafa5_dataset_name,
        cache_dir=args.dataset_cache_dir,
        dataset_subset=args.cafa5_dataset_subset,
        max_length=args.max_length_protein,
        seed=args.seed,
        val_split_ratio=args.val_split_ratio,
        return_as_chat_template=True,
        split_go_aspects=args.split_go_aspects,
        structure_dir=args.structure_dir,
        include_go_defs=args.include_go_defs,
        interpro_dataset_name=args.interpro_dataset_name,
        include_protein_function_summary=args.include_protein_function_summary,
        interpro_in_prompt=args.interpro_in_prompt,
        ppi_in_prompt=args.ppi_in_prompt,
        debug=args.debug,
    )

    n = len(val_ds) if args.max_samples <= 0 else min(args.max_samples, len(val_ds))
    samples = val_ds.select(range(n))
    samples = add_structures_to_dataset(samples, max_length_protein=args.max_length_protein)
    print(f"📊 {len(samples)} samples loaded")

    batches = build_batches(
        samples=samples,
        batch_size=args.request_batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
    )

    if batches:
        with open(args.first_batch_out, "w") as f:
            json.dump(batches[0], f, indent=2)
        print(f"💾 Saved first batch payload → {args.first_batch_out}")

    t0 = time.time()
    connector = aiohttp.TCPConnector(limit=args.concurrent_requests)
    async with aiohttp.ClientSession(connector=connector, json_serialize=_dumps) as sess:
        sem = asyncio.Semaphore(args.concurrent_requests)

        async def worker(payload):
            async with sem:
                return await _post(sess, f"{server}/generate/", payload, args.client_timeout_sec)

        tasks = [asyncio.create_task(worker(p)) for p in batches]
        done, _ = await asyncio.wait(tasks)
        results = [t.result() for t in done]

    dt = time.time() - t0
    print(f"⏱️  {len(samples)} samples | {dt:.2f}s | {len(samples)/dt:.2f} samples/s")

    if args.save_results:
        out = {
            "config": vars(args),
            "num_samples": len(samples),
            "time_sec": dt,
            "results": results,
        }
        with open(args.results_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"💾 Saved results → {args.results_out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Async CAFA inference against vLLM server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)

    # Dataset options (aligned with training)
    p.add_argument("--cafa5_dataset", type=str, default="wanglab/cafa5")
    p.add_argument("--cafa5_dataset_name", type=str, default="cafa5_reasoning")
    p.add_argument("--cafa5_dataset_subset", type=str, default=None)
    p.add_argument("--dataset_cache_dir", type=str, default="/large_storage/goodarzilab/bioreason/data/")
    p.add_argument("--structure_dir", type=str, default="/large_storage/goodarzilab/bioreason/data/sequences/")
    p.add_argument("--include_go_defs", type=str2bool, default=False)
    p.add_argument("--interpro_dataset_name", type=str, default="interpro_metadata")
    p.add_argument("--split_go_aspects", type=str2bool, default=True)
    p.add_argument("--interpro_in_prompt", type=str2bool, default=True)
    p.add_argument("--ppi_in_prompt", type=str2bool, default=True)
    p.add_argument("--include_protein_function_summary", type=str2bool, default=True)
    p.add_argument("--val_split_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--debug", type=str2bool, default=False)

    # Backward-compatibility aliases
    p.add_argument("--cafa5_hf_repo", type=str, default=None)
    p.add_argument("--cafa5_config", type=str, default=None)
    p.add_argument("--interpro_config", type=str, default=None)

    # Eval controls
    p.add_argument("--max_samples", type=int, default=128)
    p.add_argument("--max_length_protein", type=int, default=500)
    p.add_argument("--request_batch_size", type=int, default=16)
    p.add_argument("--concurrent_requests", type=int, default=8)

    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.0)

    p.add_argument("--save_results", action="store_true")
    p.add_argument("--first_batch_out", type=str, default="batches_0.json")
    p.add_argument("--results_out", type=str, default="cafa_vllm_results.json")
    p.add_argument("--client_timeout_sec", type=int, default=1800)

    args = p.parse_args()

    # Normalize legacy flags to the new names if provided
    if getattr(args, "cafa5_hf_repo", None):
        args.cafa5_dataset = args.cafa5_hf_repo
    if getattr(args, "cafa5_config", None):
        args.cafa5_dataset_name = args.cafa5_config
    if getattr(args, "interpro_config", None):
        args.interpro_dataset_name = args.interpro_config

    asyncio.run(main(args))


