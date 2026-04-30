"""Collect benchmark results from trackio for runs filtered by job ID range."""

import sys

from trackio.sqlite_storage import SQLiteStorage


def get_run_results(project: str, run_name: str, after_timestamp: str | None = None):
    """Get final MFU/TPS and peak GPU memory for a run, optionally filtered by timestamp."""
    train_logs = SQLiteStorage.get_logs(project, run_name)
    sys_logs = SQLiteStorage.get_system_logs(project, run_name)

    if after_timestamp:
        train_logs = [log for log in train_logs if log.get("timestamp", "") > after_timestamp]
        sys_logs = [log for log in sys_logs if log.get("timestamp", "") > after_timestamp]

    if not train_logs:
        return None

    # Get last training log with mfu (skip empty stats logs)
    final_train = None
    for log in reversed(train_logs):
        if log.get("train/mfu") is not None:
            final_train = log
            break
    if final_train is None:
        final_train = train_logs[-1]

    # Peak allocated GPU memory across all GPUs
    peak_mem = 0.0
    for log in sys_logs:
        for k, v in log.items():
            if k.startswith("gpu/") and "/allocated_memory" in k and isinstance(v, (int, float)):
                if v > peak_mem:
                    peak_mem = v

    # Total memory for percentage
    total_mem = 80.0
    for log in sys_logs:
        v = log.get("gpu/0/total_memory")
        if isinstance(v, (int, float)):
            total_mem = v
            break

    return {
        "mfu": final_train.get("train/mfu"),
        "tps": final_train.get("train/train_tokens_per_second"),
        "loss": final_train.get("train/loss"),
        "peak_mem_gb": peak_mem,
        "peak_mem_pct": 100 * peak_mem / total_mem if total_mem else 0,
        "total_mem_gb": total_mem,
        "n_train_logs": len(train_logs),
        "n_sys_logs": len(sys_logs),
    }


if __name__ == "__main__":
    project = "trl-sft-benchmark"
    after = sys.argv[1] if len(sys.argv) > 1 else "2026-04-24T22:00:00"

    runs_to_collect = [
        "qwen3_30b_a3b_ctx16k_n2_fsdp2_dp16_tp1_pp1_cp1_ep1",
        "qwen3_30b_a3b_ctx16k_n2_fsdp2_dp16_tp1_pp1_cp1_ep1_sonicmoe",
        "qwen3_30b_a3b_ctx16k_n2_fsdp2_dp16_tp1_pp1_cp1_ep1_vllm-flash-attn3_sonicmoe",
        "qwen3_30b_a3b_ctx32k_n2_deepspeed_zero3_dp8_tp1_pp1_cp1_ep1_sp2_vllm-flash-attn3_sonicmoe",
    ]

    print(f"Filtering by timestamp > {after}\n")
    for run in runs_to_collect:
        r = get_run_results(project, run, after_timestamp=after)
        if r is None:
            print(f"{run}: NO DATA")
            continue
        tps = r["tps"]
        try:
            tps_str = f"{float(tps):.0f}" if tps else "—"
        except (ValueError, TypeError):
            tps_str = str(tps) if tps else "—"
        print(f"{run}:")
        print(f"  MFU: {r['mfu']}%   TPS: {tps_str}   Loss: {r['loss']}")
        print(f"  Peak GPU mem: {r['peak_mem_gb']:.1f} GB ({r['peak_mem_pct']:.0f}%)")
        print(f"  ({r['n_train_logs']} train logs, {r['n_sys_logs']} sys logs)")
        print()
