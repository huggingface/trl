# DataLoader + Expert Parallelism: a study primer

A walkthrough of how distributed dataloaders work, why expert parallelism breaks them, and how the fix in `sft_trainer.py` restores the invariant.

If you've never thought hard about distributed data loading, start at the top. If you already know how `DistributedSampler` works, jump to "Adding expert parallelism to the picture".

---

## 0. The setup we'll use as a running example

- 2 nodes × 8 GPUs = **16 GPUs total**, "world size" = 16. Global ranks 0..15.
- Model is Qwen3-30B-A3B — a Mixture-of-Experts model with 128 experts, top-k=8.
- We want **expert parallelism with size 8**: split the 128 experts across 8 GPUs (16 experts per GPU).
- Total GPUs (16) ÷ EP size (8) = **2 EP groups**.
- Each EP group has 8 GPUs that share the load of computing experts.
- This means we have effectively **2 data parallel "slots"**: every step, the model processes 2 micro-batches in parallel — one per EP group.

Our config calls this `dp=16, ep=8`, but the **effective DP** (number of independent micro-batches per step) is `world_size / ep_size = 2`, not 16. This subtlety is the source of all our pain.

---

## 1. What a DataLoader does in single-GPU training

A `DataLoader` wraps a `Dataset` (a list-like object indexed by integer). On every iteration it:

1. Asks a **`Sampler`** for a list of dataset indices (e.g. `[42, 17, 99, ...]`).
2. Reads `dataset[42]`, `dataset[17]`, `dataset[99]`, ... — these are individual **samples**.
3. Hands the batch of samples to a **`collate_fn`** which stacks them into tensors (padding, packing, etc.).
4. Yields the batched tensor dict to the training loop.

The sampler decides **WHICH** indices. Common choices:
- `SequentialSampler`: iterate `[0, 1, 2, ..., N-1]`.
- `RandomSampler`: shuffle `[0..N-1]` and iterate.

In single-GPU training, that's it. Every step the model sees one batch.

---

## 2. The distributed case: making N GPUs each see different data

In data-parallel training with N GPUs, every GPU has a copy of the model. Each GPU processes a **different** batch in parallel. After backward, the gradients are all-reduced across the N GPUs so all copies stay in sync.

For this to work, we need **disjoint shards** of the dataset assigned to each GPU. If we have 100 samples and 4 GPUs:

- GPU 0 sees indices 0, 4, 8, 12, ..., 96
- GPU 1 sees indices 1, 5, 9, 13, ..., 97
- GPU 2 sees indices 2, 6, 10, 14, ..., 98
- GPU 3 sees indices 3, 7, 11, 15, ..., 99

This is what PyTorch's `DistributedSampler` does. It takes `(num_replicas=4, rank=k)` and yields the k-th stripe.

Pictorially, with 4 GPUs:

```
Dataset:          [s0, s1, s2, s3, s4, s5, s6, s7, ...]
                   |   |   |   |   |   |   |   |
GPU 0 reads:       ✓               ✓
GPU 1 reads:           ✓               ✓
GPU 2 reads:               ✓               ✓
GPU 3 reads:                   ✓               ✓
```

Every sample is read by exactly one GPU per epoch. After the optimizer step, gradients all-reduce and the 4 model copies converge. This is "pure DP".

---

## 3. Variants: TP, CP, FSDP — what changes about data sharding?

Different parallelism strategies change **what each GPU computes**, but not all of them require unique data per GPU. The dataloader has to know which dimensions are "data-shard" (different data per rank) vs "data-replicate" (same data, parallelism is over something else).

| Strategy | Same data per rank? | What's parallelized over the dim |
|---|---|---|
| **DP** (data parallel) | NO | a different micro-batch per rank |
| **FSDP** (`dp_shard`) | NO | a different micro-batch per rank, but model weights are sharded too |
| **HSDP replicate** (`dp_replicate`) | NO | a different micro-batch per replicate group |
| **TP** (tensor parallel) | YES | weight matrices are split column/row-wise |
| **CP** (context parallel) | YES | the sequence is split across ranks |
| **SP** (sequence parallel, ALST) | YES | sequence-level parallelism |
| **EP** (expert parallel) | YES | the *experts* are split across ranks |

The crucial split:
- **Data-shard dims (DP, FSDP, HSDP-replicate)**: every rank gets its own micro-batch.
- **Data-replicate dims (TP, CP, SP, EP)**: all ranks within the dim see the **same** micro-batch.

For a multi-dim setup like `(dp=2, tp=4)` with 8 GPUs, the dataloader must produce **2 unique micro-batches**, each replicated to **4 GPUs**. So 8 GPUs but only 2 distinct batches per step.

The way you tell the dataloader this is by saying:
- `effective_num_processes = product of data-shard dim sizes` (2 in the example).
- `effective_process_index = global_rank // (product of data-replicate dim sizes)` (rank 0..3 → 0, rank 4..7 → 1).

The dataloader then strides as if there were only 2 ranks, and each batch is read by 4 GPUs simultaneously.

---

## 4. Adding expert parallelism to the picture

EP works like this: instead of every GPU storing all 128 experts, each rank in the EP group stores `128 / ep_size = 16` experts. When a token's router decides "send this to expert 73", expert 73 lives on `rank 73 // 16 = 4` of the EP group. Every rank computes its own local experts' contribution to the output, then an **all-reduce sums the partials** across the 8 ranks of the EP group, producing the final output.

Critically: **all 8 ranks of an EP group receive the same `hidden_states` tensor as input.** They have to — they're computing different parts of the SAME forward pass. If the inputs differed across ranks, the partial outputs would correspond to different tokens, and summing them would be nonsense.

For data-loading purposes, EP is therefore a **data-replicate** dim: the 8 ranks of one EP group must see the same micro-batch.

In our 16-GPU `dp=16, ep=8` setup:

```
World ranks:     0  1  2  3  4  5  6  7  | 8  9  10 11 12 13 14 15
EP group:        ────── group A ──────── | ────── group B ────────
EP local rank:   0  1  2  3  4  5  6  7  | 0  1  2  3  4  5  6  7

Effective DP rank (= world_rank // ep_size):
                 0  0  0  0  0  0  0  0  | 1  1  1  1  1  1  1  1
```

So the dataloader should treat this as **2 effective ranks** (DP rank 0 and DP rank 1), each replicated to 8 GPUs.

Pictorially:

```
Dataset:          [batch0, batch1, batch2, batch3, ...]
                      |       |       |       |
EP group A reads:     ✓               ✓                ← all 8 ranks read same batch
                  (ranks 0–7)     (ranks 0–7)
EP group B reads:             ✓               ✓
                          (ranks 8–15)    (ranks 8–15)
```

8 GPUs of group A all read `batch0` together → identical input tensors → EP all-reduce happy.

---

## 5. What accelerate's `prepare_data_loader` actually does

When the trainer calls `self.accelerator.prepare(dataloader)`, the call eventually reaches `accelerate/data_loader.py::prepare_data_loader`. This function:

1. Reads `num_processes` and `process_index` (defaults: `state.num_processes` = world size; `state.process_index` = global rank).
2. Optionally **overrides** them based on `torch_device_mesh.mesh_dim_names`. The interesting block (lines 1119-1155):

```python
if torch_device_mesh:
    if state.distributed_type == DistributedType.DEEPSPEED:
        submesh_tp_size = 1
        if "tp" in torch_device_mesh.mesh_dim_names:
            submesh_tp_size = torch_device_mesh["tp"].size()
        process_index = process_index // submesh_tp_size
        num_processes  = num_processes  // submesh_tp_size
    else:
        # the FSDP/multi-GPU path
        if "tp" in torch_device_mesh.mesh_dim_names: submesh_tp_size = ...
        if "cp" in torch_device_mesh.mesh_dim_names: submesh_cp_size = ...
        if "dp_replicate" in torch_device_mesh.mesh_dim_names: submesh_dp_size = ...
        if "dp_shard" in torch_device_mesh.mesh_dim_names: submesh_fsdp_size = ...
        process_index = process_index // (submesh_tp_size * submesh_cp_size)
        num_processes = submesh_fsdp_size * submesh_dp_size
```

This is where the dataloader gets told "TP/CP are data-replicate, divide them out; dp_replicate/dp_shard are data-shard, multiply them in to get the effective shard count".

3. Builds a `BatchSamplerShard` with the (possibly-overridden) `num_processes` and `process_index`. This is the thing that actually controls which dataset indices each rank reads.

4. Wraps the original dataloader in a `DataLoaderShard` (or `DataLoaderDispatcher`) that uses the new batch sampler.

The `BatchSamplerShard` algorithm: round-robin distribute batches across `num_processes`. Rank `process_index` reads batches `process_index, process_index + num_processes, process_index + 2*num_processes, …`.

---

## 6. The bug: accelerate has never heard of EP

Look at the mesh-name list above:

> `tp`, `cp`, `dp_replicate`, `dp_shard`

Notice what's missing: **`ep`**. Accelerate's `ParallelismConfig` has fields for `tp_size`, `cp_size`, `dp_replicate_size`, `dp_shard_size`, `sp_size` — **no `ep_size`**. EP isn't a first-class concept anywhere in accelerate. The code does not divide by EP size when computing `process_index`. So every world rank gets a unique data shard, even though within an EP group they should all see the same batch.

Worse, transformers builds an EP device mesh **internally** in `tensor_parallel.py::apply_tp_plan` to drive the EP all-reduces. That mesh is never registered with accelerate. So even if accelerate knew to look for `"ep"` in the mesh names, the mesh accelerate sees doesn't have it.

End state in our 16-GPU run:
- accelerate's prepare gets `num_processes=16, process_index=0..15`.
- It builds 16 unique data shards.
- Every world rank reads a different micro-batch.
- Inside the model forward, the EP layers all-reduce 8 ranks at a time.
- The 8 ranks of an EP group have **different `(seq_len, hidden)` shapes**, because their micro-batches packed to different lengths (we use `--packing --pad_to_multiple_of 1`, so sequence length varies per packed batch).
- NCCL all-reduce silently hangs when participant tensor shapes differ (state = `scheduled, never started`).

The hang is **probabilistic-but-deterministic**: most batches happen to pack to exactly `max_length`, so the 8 ranks coincidentally agree. Once a single rank's batch packs to something different, the EP all-reduce hangs forever and the watchdog times out 600 s later.

Concretely from job `22102169`, step 16 of training:

| World rank | EP group | input_ids shape sent to experts |
|---|---|---|
| 0,1,2,3,5,6,7 | A | `[1, 65536]` (7 ranks agree) |
| **4** | **A** | **`[1, 54223]`** (mismatch!) |
| 8..15 | B | `[1, 65536]` (all 8 agree) |

Group A's all-reduce can't start (shapes don't match). 7 ranks fire watchdog. Rank 4 doesn't because its all-reduce launched a different-sized one and was waiting on a phantom 1-rank collective.

---

## 7. The fix: tell `prepare_data_loader` what the effective DP count is

The proper architectural fix is two-sided: add `ep_size` to accelerate's `ParallelismConfig` and to the mesh-inspection block at line 1119, and have transformers populate it. Until those upstream PRs land, the local fix sits entirely in TRL's `SFTTrainer`.

The plan: bypass accelerate's mesh inspection by calling `prepare_data_loader` directly with EP-corrected `num_processes` and `process_index`.

The implementation: override `_get_dataloader` (the trainer hook that builds the dataloader). When EP is enabled, monkey-patch `accelerator.prepare_data_loader` for **one call**, replacing it with a version that:
- computes `eff_num = world_size // ep_size` (= 2)
- computes `eff_idx = global_rank // ep_size` (= 0 for ranks 0-7, 1 for ranks 8-15)
- calls the underlying `accelerate.data_loader.prepare_data_loader` directly with those values
- passes `torch_device_mesh=None` so the mesh-inspection block at line 1119-1155 doesn't override our values
- leaves all other behavior untouched (rng, dispatch, even_batches, stateful, …).

After `super()._get_dataloader(...)` returns, the patch is removed. Subsequent accelerate operations (model prep, optimizer prep) are unaffected.

The actual code (see `trl/trainer/sft_trainer.py:1568`):

```python
def _get_dataloader(self, dataset, description, batch_size, sampler_fn=None, is_training=False, dataloader_key=None):
    if not getattr(self.args, "enable_expert_parallel", False):
        return super()._get_dataloader(dataset, description, batch_size, sampler_fn, is_training, dataloader_key)

    from accelerate.data_loader import prepare_data_loader as _prep_dl

    ep_size = getattr(self.args, "expert_parallel_size", None) or 1
    eff_num = self.accelerator.num_processes // ep_size
    eff_idx = self.accelerator.process_index // ep_size

    orig = self.accelerator.prepare_data_loader

    def _patched(dataloader, device_placement=None, slice_fn_for_dispatch=None):
        if device_placement is None:
            device_placement = self.accelerator.device_placement
        return _prep_dl(
            dataloader,
            self.accelerator.device,
            num_processes=eff_num,
            process_index=eff_idx,
            split_batches=self.accelerator.split_batches,
            put_on_device=device_placement,
            rng_types=self.accelerator.rng_types.copy(),
            dispatch_batches=self.accelerator.dispatch_batches,
            even_batches=self.accelerator.even_batches,
            slice_fn_for_dispatch=slice_fn_for_dispatch,
            use_seedable_sampler=self.accelerator.use_seedable_sampler,
            data_seed=self.accelerator.dataloader_config.data_seed,
            non_blocking=self.accelerator.non_blocking,
            use_stateful_dataloader=self.accelerator.use_stateful_dataloader,
            torch_device_mesh=None,
        )

    self.accelerator.prepare_data_loader = _patched
    try:
        return super()._get_dataloader(dataset, description, batch_size, sampler_fn, is_training, dataloader_key)
    finally:
        self.accelerator.prepare_data_loader = orig
```

After this:
- `BatchSamplerShard(num_processes=2, process_index=eff_idx)` strides through the dataset in chunks of 2 batches per rank.
- All 8 ranks within EP group A share `eff_idx=0` → they read identical sequences of dataset indices → identical input tensors.
- Same for group B with `eff_idx=1`.
- The experts forward sees agreeing `(seq_len, hidden)` shapes across the 8 ranks → NCCL all-reduce completes normally.

Verified empirically by Test G (job `22102668`): the same config that hung deterministically at step 16 in Test D now completes all 30 steps cleanly with healthy loss and gradients.

---

## 8. The proper upstream fix (for when we have time)

Make `ep` a first-class concept in accelerate, mirroring `tp`:

**`accelerate/parallelism_config.py`** — add `ep_size` field:
```python
ep_size: Optional[int] = None
```

**`accelerate/data_loader.py:1146-1155`** — add `submesh_ep_size`:
```python
submesh_ep_size = 1
if "ep" in torch_device_mesh.mesh_dim_names:
    submesh_ep_size = torch_device_mesh["ep"].size()
process_index = process_index // (submesh_tp_size * submesh_cp_size * submesh_ep_size)
```
And the equivalent in the DeepSpeed branch (line 1119).

**`transformers/trainer.py`** — when `model.config.enable_expert_parallel` is set, push `ep_size` into `accelerate.state.parallelism_config.ep_size` so the mesh-inspection block above sees it.

Once these three changes land, delete the local `_get_dataloader` override in TRL — it becomes redundant.

---

## 9. Recap as a checklist

If you're debugging a similar bug in another stack, here's the mental model to apply:

- [ ] Identify each parallelism dim in your stack (DP, FSDP, TP, CP, SP, EP, PP, …).
- [ ] For each dim, classify it as **data-shard** (different data per rank in the dim) or **data-replicate** (same data per rank in the dim).
- [ ] Compute `effective_num_processes = product(data-shard dim sizes)`.
- [ ] Compute `effective_process_index` such that ranks within the same data-replicate group share the same value.
- [ ] Verify your dataloader's sampler is striding with those effective values, not with `world_size` and `global_rank`.
- [ ] Add a per-rank assertion right before any cross-rank collective: tensors at the collective's input must have identical shape across the participants. If not, the dataloader is wrong.

The bug we hit was: EP wasn't classified as data-replicate anywhere in accelerate, so the shape invariant at the EP all-reduce was being violated. The fix made the EP dim explicit in the dataloader's view of the world.
