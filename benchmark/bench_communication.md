# NCCL Communication Benchmark

## Hardware

- **Instance**: AWS p5.48xlarge
- **GPUs**: 8× H100 SXM 80GB per node
- **Intra-node**: NVLink (900 GB/s bidirectional between GPUs)
- **Inter-node**: 32× EFA (Elastic Fabric Adapter) NICs, 100 Gbps each = **3200 Gbps (400 GB/s) aggregate per node**
- **No InfiniBand** — EFA uses RDMA over a custom AWS network fabric

## Bandwidth metrics

Two bandwidth metrics are reported, following the [NCCL-tests convention](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md):

**Algorithm Bandwidth (AlgBW)** = `data_size / time`. Measures how fast the operation completes from the application's perspective. `data_size` is the per-rank input size for allreduce and all_to_all, or total output size for allgather, or total input size for reduce_scatter.

**Bus Bandwidth (BusBW)** = AlgBW × correction factor. Estimates the actual bandwidth on the hardware links by accounting for the communication pattern:
- **allreduce**: `AlgBW × 2(N-1)/N` — ring sends data twice around (reduce + broadcast)
- **allgather, reduce_scatter, all_to_all**: `AlgBW × (N-1)/N` — ring sends data once

BusBW is useful for comparing against hardware specs (e.g., NVLink 900 GB/s, EFA 400 GB/s). A BusBW close to the link bandwidth means the collective is saturating the hardware.

**Caveat**: BusBW is designed for ring-based collectives. For all_to_all (which is not ring-based), the formula underestimates actual link utilization because it doesn't account for the mixed intra/inter-node traffic pattern. See the analysis section below.

## Benchmark

Script: `benchmark/nccl_bench.py`

Tests four collective operations at message sizes from 1MB to 1GB, using the flat tensor APIs (`all_gather_into_tensor`, `reduce_scatter_tensor`). 5 warmup + 20 timed iterations per measurement.

### Results at 1GB message size

#### 1 node (8 GPUs) — NVLink only

| Op             | Time (ms) | AlgBW (GB/s) | BusBW (GB/s) |
| -------------- | --------- | ------------ | ------------ |
| allreduce      | 4.17      | 257.6        | 450.9        |
| allgather      | 20.88     | 411.3        | 359.9        |
| reduce_scatter | 20.84     | 412.2        | 360.7        |
| all_to_all     | 2.82      | 380.7        | 333.1        |

All intra-node ops achieve 330-450 GB/s bus bandwidth, consistent with NVLink.

#### 2 nodes (16 GPUs) — NVLink intra-node + EFA inter-node

| Op             | Time (ms) | AlgBW (GB/s) | BusBW (GB/s) |
| -------------- | --------- | ------------ | ------------ |
| allreduce      | 4.87      | 220.7        | 413.7        |
| allgather      | 57.11     | 300.8        | 282.0        |
| reduce_scatter | 54.87     | 313.1        | 293.5        |
| all_to_all     | 27.14     | 39.6         | 37.1         |

#### Inter-node vs intra-node ratio

| Op             | 1 node BusBW | 2 node BusBW | Ratio   | Why                                                      |
| -------------- | ------------ | ------------ | ------- | -------------------------------------------------------- |
| allreduce      | 451 GB/s     | 414 GB/s     | **92%** | Ring/tree algorithm pipelines across all 32 EFA NICs     |
| allgather      | 360 GB/s     | 282 GB/s     | **78%** | Ring-based gather, scales reasonably                     |
| reduce_scatter | 361 GB/s     | 294 GB/s     | **81%** | Ring-based reduce, similar to allgather                  |
| all_to_all     | 333 GB/s     | 37 GB/s      | **11%** | N×N personalized exchange — cannot use ring optimization |

## Understanding the numbers

### allreduce vs allgather/reduce_scatter time

allreduce (4.17ms) is faster than allgather (20.88ms) despite both being ring-based because of data size:
- **allreduce**: 1GB in, 1GB out (same size)
- **allgather**: 1GB in per rank, 8GB out total (8× more data movement with 8 GPUs)

allreduce = reduce_scatter + allgather pipelined in a ring, but operates on the **same-sized** tensor. allgather moves N× more total data.

### Why ring-based ops scale but all_to_all doesn't

**allreduce, allgather, reduce_scatter** use NCCL's ring algorithm:

- Data flows around a ring: each GPU sends to one neighbor, receives from another
- At any moment, each EFA NIC has exactly one send + one receive in flight
- All 32 NICs work in parallel on different ring segments → near-full utilization
- Inter-node scaling: 78-92% of intra-node bandwidth

**all_to_all** is a personalized exchange:

- Each of 8 GPUs on node A must send unique data to each of 8 GPUs on node B
- With 16 GPUs across 2 nodes: 50% of traffic is intra-node (NVLink, fast), 50% is inter-node (EFA, slow)
- The inter-node portion: 4 GB per direction, completed in 27ms → **147 GB/s per direction** (37% of 400 GB/s EFA capacity)
- No ring/tree optimization possible — every transfer has a unique source-destination pair

### Correcting the busbw comparison

The busbw formula `algbw × (N-1)/N` is designed for ring-based ops and is misleading for all_to_all. A fairer comparison of actual inter-node utilization:

| Op | Inter-node BW achieved | EFA capacity | Utilization |
|---|---|---|---|
| allreduce | ~300 GB/s bidirectional | 400 GB/s | ~75% |
| allgather | ~240 GB/s bidirectional | 400 GB/s | ~60% |
| reduce_scatter | ~250 GB/s bidirectional | 400 GB/s | ~63% |
| all_to_all | ~295 GB/s bidirectional | 400 GB/s | ~37% per direction |

The all_to_all inter-node gap is **~2x** (37% vs 60-75% utilization), not 9x as the raw busbw numbers suggest. The 9x ratio in busbw comes from the formula not accounting for all_to_all's mixed intra/inter-node traffic pattern.

## Impact on MoE training

| Strategy             | Key collectives            | 1-node BusBW   | 2-node BusBW   | Training impact            |
| -------------------- | -------------------------- | -------------- | -------------- | -------------------------- |
| FSDP2 param sharding | allgather + reduce_scatter | 360 + 361 GB/s | 282 + 294 GB/s | ~80% scaling inter-node    |
| EP token routing     | all_to_all                 | 333 GB/s       | 37 GB/s        | **9x slowdown inter-node** |
| CP ring attention    | P2P ring (allreduce-like)  | ~451 GB/s      | ~414 GB/s      | 92% scaling                |
| Gradient sync        | allreduce                  | 451 GB/s       | 414 GB/s       | 92% scaling                |

This explains all observed MFU behavior:

- **EP MFU peaks at 2 nodes** (4.77%) and degrades at 8 nodes (3.55%) — all_to_all is 9x slower inter-node
- **FSDP2 scales reasonably** — allgather/reduce_scatter at 78-81% inter-node ratio
- **CP works well inter-node** — ring attention uses allreduce-like patterns (92% scaling)
- **Optimal strategy on this cluster**: keep EP intra-node (1 node), use DP+CP for cross-node scaling
