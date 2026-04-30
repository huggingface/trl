# FSDP vs DeepSpeed

Accelerate offers flexibilty of training frameworks, by integrating two extremely powerful tools for distributed training, namely [Pytorch FSDP](../usage_guides/fsdp) and [Microsoft DeepSpeed](../usage_guides/deepspeed). The aim of this tutorial is to draw parallels, as well as to outline potential differences, to empower the user to switch seamlessly between these two frameworks.

To switch between the frameworks, we recommend launching code `accelerate launch` passing in the correct config file with `--config_file`, or passing in the respective arguments directly for [FSDP and DeepSpeed](../package_reference/cli#accelerate-launch) .

Example Accelerate configurations can be found here for [DeepSpeed](../usage_guides/deepspeed#accelerate-deepspeed-plugin) and [FSDP](../usage_guides/fsdp#how-it-works-out-of-the-box), or in the [example zoo under "Launch Configurations"](../usage_guides/explore)

This tutorial is for single-node, multi-GPU, scenarios only.

## Configuring Functionalities

Model tensors are split into different GPUs in an attempt to scale up model sizes; this is termed _sharding_ in FSDP, and _partitioning_ in DeepSpeed. FSDP sharding and DeepSpeed ZeRO (partitioning) stages are configured by `--fsdp_sharding_strategy`, and `--zero_stage`, respectively. In particular, FSDP `FULL_SHARD` maps to DeepSpeed ZeRO stage `3`; see this [comprehensive mapping between FSDP sharding and DeepSpeed ZeRO settings](../usage_guides/fsdp#mapping-between-fsdp-sharding-strategies-and-deepspeed-zero-stages). The below table summarizes and groups similar settings:

| Group                   | Framework     | Configuration                                                               | Example                    | Restrictions (if any)                           |
| ----------------------- | ------------- | --------------------------------------------------------------------------- | -------------------------- | ----------------------------------------------- |
| sharding / partitioning | FSDPDeepSpeed | `--fsdp_sharding_strategy``--zero_stage`                                    | `1` (`FULL_SHARD`) `3`     |
| offload                 | FSDPDeepSpeed | `--fsdp_offload_params``--offload_param_device``--offload_optimizer_device` | `true``cpu``cpu`           | all or nothing                                  |
| model loading           | FSDPDeepSpeed | `--fsdp_cpu_ram_efficient_loading``--zero3_init_flag`                       | `true``true`               | only ZeRO 3                                     |
| efficient checkpointing | FSDPDeepSpeed | `--fsdp_state_dict_type``--zero3_save_16bit_model`                          | `SHARDED_STATE_DICT``true` | only ZeRO 3                                     |
| weights prefetching     | FSDPDeepSpeed | `--fsdp_forward_prefetch``--fsdp_backward_prefetch`None                     | `true``BACKWARD_PRE`       |
| model                   | FSDPDeepSpeed | `--fsdp_auto_wrap_policy``--fsdp_transformer_layer_cls_to_wrap`None         | `TRANSFORMER_BASED_WRAP`   | Usually not needed Transparent to user.         |
| parameters summoning    | FSDPDeepSpeed | `--fsdp_use_orig_params`None                                                | `true`                     | required for `torch.compile`Transparent to user |
| parameters syncing      | FSDPDeepSpeed | `--fsdp_sync_module_states`None                                             | `true`                     |
| training                | FSDPDeepSpeed | None`--gradient_accumulation_steps``--gradient_clipping`                    | `auto``auto`               | Transparent to user                             |

For detailed descriptions of the above, refer to [`Accelerate` launch documentation](../package_reference/cli#accelerate-launch).

    To access other DeepSpeed configurations, such as mixed precision settings,
    you need to pass in a `--deepspeed_config_file`, see the [documentation](../usage_guides/deepspeed#deepspeed-config-file).

    DeepSpeed can be also configured via [DeepSpeedPlugin](/docs/accelerate/v1.13.0/en/package_reference/deepspeed#accelerate.DeepSpeedPlugin), e.g., `DeepSpeedPlugin.zero_stage` is equivalent of `--zero_stage`, and `DeepSpeedPlugin.hf_ds_config` can be used to pass `--deepeed_config_file.`

    FSDP can be also configured via [FullyShardedDataParallelPlugin](/docs/accelerate/v1.13.0/en/package_reference/fsdp#accelerate.FullyShardedDataParallelPlugin), e.g., `FullyShardedDataParallelPlugin.sharding_strategy` is equivalent of `--fsdp_sharding_strategy`.

### Checkpointing

Do note that while FSDP can be configured via `--fsdp_state_dict_type` to save either full / sharded checkpoints.

    For DeepSpeed Zero3, one could pass a `--zero3_save_16bit_model true`, which conveniently consolidates the model to a single rank and saves; this is the FSDP equivalent of `fsdp_state_dict_type: FULL_STATE_DICT`.

    For large models, consolidating the model to a single rank can be very slow.

    For quicker checkpointing, for FSDP use `fsdp_state_dict_type: SHARDED_STATE_DICT`, and for DeepSpeed Zero3 [use the `zero_to_fp32.py` script to post-convert sharded checkpoints](https://www.deepspeed.ai/tutorials/zero/#extracting-weights).

### Offloading

FSDP only allows _all-or-nothing_ offload (i.e., either offload parameters, gradients, and optimizer, or keep them all in GPU), but DeepSpeed can offload parameters and optimizer differently. Furthermore, DeepSpeed also supports [offloading to NVME](https://www.deepspeed.ai/docs/config-json/#parameter-offloading).

### Prefetching

FSDP allows two prefetching configurations `--fsdp_forward_prefetch` and `--fsdp_backward_prefetch` to improve overlap of comms / computation at a cost of extra memory, see [FSDP documentation](https://pytorch.org/docs/stable/fsdp.html).
For DeepSpeed, the prefetching will be turned on when needed, and it turns on depending on certain hyper-params like `stage3_param_persistence_threshold`, `stage3_max_reuse_distance`, etc, [that can be configured for Zero3](https://www.deepspeed.ai/docs/config-json/#parameter-offloading); `accelerate` may set these hyper-params automatically if you don't set those explicitly in the deepspeed config file.

    For FSDP set `fsdp_backward_prefetch: BACKWARD_PRE` for improved throughputs if memory allows.

### Model Loading

While FSDP require an explicit `--fsdp_cpu_ram_efficient_loading true` to activate efficient model loading, `transformers` will activate the similar feature whenever DeepSpeed Zero3 is used.

    For FSDP, whenever setting `--fsdp_cpu_ram_efficient_loading true`, `accelerate` will automatically set `sync_module_states` to true.
    For RAM efficient loading the weights will be loaded only in a single rank, and thus requires `sync_module_states` to broadcast weights to other ranks.

### Model

FSDP requires an explicit `--fsdp_auto_wrap_policy` for the algorithm to decide how to schedule the all-gather and reduce-scatter operations. But for DeepSpeed this is transparent to the user.

    For FSDP, simply set `fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP`. With the latest `transformers` versions, we try our best to figure out the suitable `fsdp_transformer_layer_cls_to_wrap` for HF transformers models. However, if you get an error regarding it, please specify this.

### Parameters Summoning

FSDP requires an explicit `--fsdp_use_orig_params` flag if using `torch.compile`, see [the pytorch documentation](https://pytorch.org/docs/stable/fsdp.html#module-torch.distributed.fsdp). For DeepSpeed this is transparent to the user.

    For FSDP, when using `torch.compile` please set `fsdp_use_orig_params: True`.

## Training

Deepspeed requires explicit `--gradient_accumulation_steps` and `--gradient_clipping` flags. For FSDP this is transparent to the user.

    When using DeepSpeed, set `gradient_accumulation_steps: "auto"` and `gradient_clipping: "auto"` to automatically pick up values set in the [Accelerator](/docs/accelerate/v1.13.0/en/package_reference/accelerator#accelerate.Accelerator) or `TrainingArguments` (if using `transformers`).

## On Differences in Data Precision Handling

To discuss how data precision is handled in both FSDP and Deepspeed, it is instructive to first give an overview of how model parameters are handled in these frameworks. Before the model / optimizer parameters are distributed across GPUs, parameter preparation is involved to first "flatten" them to one-dimensional [`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html#torch-tensor). The implementation of FSDP / DeepSpeed varies in the respect of the `dtype` in which these "flattened" parameters are stored, and there are ramifications with regards to how [`torch.Optimizer`](https://pytorch.org/docs/stable/optim.html#module-torch.optim) allocate their `dtype`s. The table below outlines the processes for both frameworks; the "Local" column indicates the process occurring at a per-gpu level, therefore any memory overheads by upcasting should be understood to be amortized by the number of gpus used.

    As a rule of thumb, for stable training with automatic mixed precision, all the trainable parameters have to be in `torch.float32`.

| Process                                                                  | Local | Framework     | Details                                                                                                                                                               |
| ------------------------------------------------------------------------ | ----- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Loading, i.e., `AutoModel.from_pretrained(..., torch_dtype=torch_dtype)` |
| Preparation, i.e., creation of "flat params"                             | ✅    | FSDPDeepSpeed | created in `torch_dtype`. disregards `torch_dtype`, created in `float32`.                                                                                             |
| Optimizer initialization                                                 | ✅    | FSDPDeepSpeed | creates parameters in `torch_dtype` creates parameters in `float32`                                                                                                   |
| Training Step, i.e, forward, backward, reduction                         |       | FSDPDeepSpeed | follows [`MixedPrecision`](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.MixedPrecision) follows `deepspeed_config_file` mixed precision settings. |
| Optimizer (Pre-Step)                                                     | ✅    | FSDPDeepSpeed | upcasting (if any) to `torch_dtype`upcasted to `float32`                                                                                                              |
| Optimizer (Actual Step)                                                  | ✅    | FSDPDeepSpeed | occurs in `torch_dtype` occurs in `float32`.                                                                                                                          |

    Therefore when using DeepSpeed a small number of GPUs, be aware of potentially significant memory overheads due to the upcasting during preparation.

    With FSDP, in the absence of mixed precision, it is possible to operate the [`torch.Optimizer`](https://pytorch.org/docs/stable/optim.html#module-torch.optim) in low precision `torch_dtype`, which may be helpful when using small number of GPUs.

    With mixed precision, FSDP and DeepSpeed will upcast in the model preparation step (c.f. table above). But do note that FSDP will then save checkpoints in the upcasted precision; Deepspeed may still save low precision checkpoints if `--zero3_save_16bit_model` is specified.

To clarify the above table consider the concrete examples below; the optimizer pre- and actual step combined for brevity. With FSDP it is possible to operate in the two modes shown below, but DeepSpeed can only operate in one.

| Framework | Model Loading (`torch_dtype`) | Mixed Precision | Preparation (Local) | Training | Optimizer (Local) |
| --------- | ----------------------------- | --------------- | ------------------- | -------- | ----------------- |
| FSDP      | bf16                          | default (none)  | bf16                | bf16     | bf16              |
| FSDP      | bf16                          | bf16            | fp32                | bf16     | fp32              |
| DeepSpeed | bf16                          | bf16            | fp32                | bf16     | fp32              |
