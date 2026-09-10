# DeepSpeed Integration

> [!WARNING]
> Section under construction. Feel free to contribute!

TRL supports training with DeepSpeed, a library that implements advanced training optimization techniques. These include optimizer state partitioning, offloading, gradient partitioning, and more.

DeepSpeed integrates the [Zero Redundancy Optimizer (ZeRO)](https://huggingface.co/papers/1910.02054), which allows to scale the model size proportional to the number of devices with sustained high efficiency.

![ZeRO Stages](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/zero_stages.png)

## Installation

To use DeepSpeed with TRL, install it using the following command:

```bash
pip install deepspeed
```

## Running Training Scripts with DeepSpeed

No modifications to your training script are required. DeepSpeed is enabled through the training arguments, like for any [`~transformers.Trainer`]: set `deepspeed` to the path of a DeepSpeed configuration file, and launch with torchrun.

```python
training_args = SFTConfig(..., deepspeed="<DEEPSPEED_CONFIG_FILE.json>")
```

```bash
torchrun --nproc_per_node 8 train.py
```

Scripts that parse their arguments with [`TrlParser`], such as the ones behind the `trl` CLI, take it from the command line:

```bash
trl sft ... --deepspeed <DEEPSPEED_CONFIG_FILE.json>
```

We provide ready-to-use DeepSpeed configuration files in the [`examples/deepspeed_configs`](https://github.com/huggingface/trl/tree/main/examples/deepspeed_configs) directory. For example, to run training with ZeRO Stage 2:

```bash
trl sft ... --deepspeed examples/deepspeed_configs/zero2.json
```

Values set to `"auto"` in these files (batch size, gradient accumulation, precision, …) are filled in from the training arguments.

## Additional Resources

Consult the 🤗 Transformers [DeepSpeed documentation](https://huggingface.co/docs/transformers/deepspeed) for more information about the DeepSpeed integration.
