#!/usr/bin/env python3
import argparse, asyncio, time, json, aiohttp
from typing import List, Dict, Any
import os
from bioreason2.utils import str2bool
from datasets import load_dataset
from trl.scripts.eval_cafa_vllm import (
    add_structures_to_dataset,
    _flatten_user_messages_to_text,
    build_batches,
)


async def _post(session: aiohttp.ClientSession, url: str, payload: Dict[str, Any], timeout_s: int):
    async with session.post(url, json=payload, timeout=timeout_s) as resp:
        resp.raise_for_status()
        return await resp.json()


async def main(args):
    # Build server endpoints
    servers = [f"http://{args.hosts}:{p}" for p in args.ports]
    print(f"Targets: {servers}")

    # Load dataset via existing eval loader flags using load_cafa5_dataset
    from bioreason2.dataset.cafa5.load import load_cafa5_dataset
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

    # Round-robin dispatch across servers
    t0 = time.time()
    connector = aiohttp.TCPConnector(limit=args.concurrent_requests)
    async with aiohttp.ClientSession(connector=connector, json_serialize=json.dumps) as sess:
        sem = asyncio.Semaphore(args.concurrent_requests)
        async def worker(idx_payload):
            idx, payload = idx_payload
            server = servers[idx % len(servers)]
            async with sem:
                # Fast compatibility: some server branches expect 'inputs' instead of 'prompts'
                if "prompts" in payload and "inputs" not in payload:
                    payload["inputs"] = payload["prompts"]
                return await _post(sess, f"{server}/generate/", payload, args.client_timeout_sec)

        tasks = [asyncio.create_task(worker((i, p))) for i, p in enumerate(batches)]
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
    p = argparse.ArgumentParser(description="Round-robin CAFA inference across multiple vLLM servers")
    p.add_argument("--hosts", type=str, default="127.0.0.1")
    p.add_argument("--ports", type=int, nargs="+", default=[8001, 8002, 8003, 8004])

    # Dataset options (aligned with eval_cafa_vllm.py / training)
    p.add_argument("--cafa5_dataset", type=str, default="wanglab/cafa5")
    p.add_argument("--cafa5_dataset_name", type=str, default="cafa5_reasoning")
    p.add_argument("--cafa5_dataset_subset", type=str, default=None)
    p.add_argument("--dataset_cache_dir", type=str, default="/large_storage/goodarzilab/bioreason/data/")
    p.add_argument("--structure_dir", type=str, default="/large_storage/goodarzilab/bioreason/data/structures/")
    p.add_argument("--include_go_defs", type=str2bool, default=False)
    p.add_argument("--interpro_dataset_name", type=str, default="interpro_metadata")
    p.add_argument("--split_go_aspects", type=str2bool, default=True)
    p.add_argument("--interpro_in_prompt", type=str2bool, default=True)
    p.add_argument("--ppi_in_prompt", type=str2bool, default=True)
    p.add_argument("--include_protein_function_summary", type=str2bool, default=True)
    p.add_argument("--val_split_ratio", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--debug", type=str2bool, default=False)

    # Eval controls
    p.add_argument("--max_samples", type=int, default=128)
    p.add_argument("--max_length_protein", type=int, default=500)
    p.add_argument("--request_batch_size", type=int, default=8)
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
    asyncio.run(main(args))


