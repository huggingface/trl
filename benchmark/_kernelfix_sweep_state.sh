#!/bin/bash
# Persistent watcher for the kernel-fix champion sweep.
# Emits one line per state transition for each tracked job:
#   <hh:mm:ss> <jobid> <state> <ctx>k<nodes>n  <outcome>
# where outcome is one of:
#   OK          — train_runtime line + final mfu_window — needs harvest
#   CLUSTER     — NCCL connection refused / NVLink fault / NODE_FAIL — resubmit candidate
#   OOM         — CUDA out of memory — do not resubmit
#   NAN         — train loss = 0 + entropy nan — do not resubmit (config bug)
#   FAIL        — other failure — investigate
#   TIMEOUT     — slurm walltime
# Each tracked job emits exactly once when it reaches a terminal state.

STATE_FILE=/tmp/kernelfix_sweep_seen.txt
mkdir -p $(dirname $STATE_FILE)
touch $STATE_FILE

# Post-E3-mitigation resubmit (2026-05-01 19:37): NCCL_HEARTBEAT_TIMEOUT_SEC=600 in launch.sh.j2.
# Skipped: P2 (reproducible step-15/16 ALLREDUCE bug — 3x failures), Q5 (legitimate OOM),
# Q9 (already validated OK at 22099370 as 38.74% peak).
JOBS=(
  # P-series (kernelfix_p1_to_p7) — fresh resubmit with E3 fix
  "22099522:P1-128k-4n-EP8"
  "22099523:P3-64k-4n-EP8"
  "22099524:P4-256k-8n-SP2-compile"
  "22099525:P5-512k-8n-SP4-compile"
  "22099526:P6-1M-8n-SP8-compile"
  "22099527:P7-128k-8n-EP8"
  # Q-series (kernelfix_q1_to_q10) — fresh resubmit with E3 fix
  "22099528:Q1-64k-8n-EP64-compile"
  "22099529:Q2-64k-4n-EP32-compile"
  "22099530:Q3-128k-4n-EP32-compile"
  "22099531:Q4-128k-4n-EP32-SP2"
  "22099532:Q6-128k-4n-EP32-noFA3"
  "22099533:Q7-128k-4n-EP32-groupedmm"
  "22099534:Q8-32k-4n-EP32"
  "22099535:Q10-128k-8n-EP64"
)

while true; do
  for entry in "${JOBS[@]}"; do
    j=${entry%%:*}
    rest=${entry#*:}
    label=${rest//:/-}

    # Skip if already emitted
    if grep -q "^$j " $STATE_FILE 2>/dev/null; then
      continue
    fi

    # slurm state
    state=$(sacct -j $j --format=State%20 --noheader -n -X 2>/dev/null | head -1 | awk '{print $1}')

    case "$state" in
      ""|RUNNING|PENDING|REQUEUED|CONFIGURING|COMPLETING)
        continue
        ;;
    esac

    # Terminal — diagnose
    LOG=$(ls /fsx/amine_dirhoussi/trl/benchmark/logs/bench-qwen3_30b_a3b_*-${j}.{out,err} 2>/dev/null)
    OUT_LOG=$(ls /fsx/amine_dirhoussi/trl/benchmark/logs/bench-qwen3_30b_a3b_*-${j}.out 2>/dev/null | head -1)
    ERR_LOG=$(ls /fsx/amine_dirhoussi/trl/benchmark/logs/bench-qwen3_30b_a3b_*-${j}.err 2>/dev/null | head -1)

    outcome="FAIL"
    if [ "$state" = "COMPLETED" ]; then
      if [ -f "$OUT_LOG" ] && grep -q "train_runtime.*train_loss" "$OUT_LOG"; then
        # Check final loss for nan
        if grep -E "(train_loss.*nan|'entropy': 'nan'.*'mean_token_accuracy': '0.0)" "$OUT_LOG" | tail -3 | grep -q nan; then
          outcome="NAN"
        else
          outcome="OK"
        fi
      else
        outcome="FAIL"
      fi
    elif [ "$state" = "OUT_OF_MEMORY" ]; then
      outcome="OOM"
    elif grep -aqE "out of memory|CUDA out of memory" "$ERR_LOG" 2>/dev/null; then
      outcome="OOM"
    elif grep -aqE "NCCL WARN.*Connection refused|Invalid access of peer GPU memory|recv_close_deferred|node fail|Watchdog caught collective operation timeout|Process group watchdog thread terminated|TCPStore server has shut down|Broken pipe" "$ERR_LOG" 2>/dev/null; then
      outcome="CLUSTER"
    elif [ "$state" = "TIMEOUT" ]; then
      outcome="TIMEOUT"
    elif [ "$state" = "NODE_FAIL" ]; then
      outcome="CLUSTER"
    fi

    # Harvest mfu_window mean+peak if OK
    extra=""
    if [ "$outcome" = "OK" ] && [ -f "$OUT_LOG" ]; then
      mfu_vals=$(grep -oE "'mfu_window': '[0-9.e+-]+'" "$OUT_LOG" | grep -oE "[0-9.e+-]+" )
      if [ -n "$mfu_vals" ]; then
        mean=$(echo "$mfu_vals" | awk '{s+=$1; n++} END {if(n>0) printf "%.2f", s/n}')
        peak=$(echo "$mfu_vals" | sort -nr | head -1)
        loss=$(grep -oE "'train_loss': '[0-9.]+'" "$OUT_LOG" | tail -1 | grep -oE "[0-9.]+")
        extra=" mean=${mean}% peak=${peak}% loss=${loss}"
      fi
    fi

    ts=$(date +%H:%M:%S)
    echo "$j $state $outcome" >> $STATE_FILE
    echo "$ts $j $label $state $outcome$extra"
  done

  # All done?
  done_count=$(wc -l < $STATE_FILE)
  total=${#JOBS[@]}
  if [ "$done_count" -ge "$total" ]; then
    echo "$(date +%H:%M:%S) ALL_DONE $done_count/$total"
    break
  fi

  sleep 60
done
