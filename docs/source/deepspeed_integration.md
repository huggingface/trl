# DeepSpeed Integration

<Tip warning={true}>

Section under construction. Feel free to contribute!

</Tip>

TRL supports training with DeepSpeed, a library that implements advanced training optimization techniques. These include optimizer state partitioning, offloading, gradient partitioning, and more.

DeepSpeed integrates the [Zero Redundancy Optimizer (ZeRO)](https://huggingface.co/papers/1910.02054), which allows to scale the model size proportional to the number of devices with sustained high efficiency.

![ZeRO Stages](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/zero_stages.png)

## Installation

To use DeepSpeed with TRL, install it using the following command:

```bash
pip install deepspeed
```

## Running Training Scripts with DeepSpeed

No modifications to your training script are required. Simply run it with the DeepSpeed configuration file:

```bash
accelerate launch --config_file <ACCELERATE_WITH_DEEPSPEED_CONFIG_FILE.yaml> train.py
```

We provide ready-to-use DeepSpeed configuration files in the [`examples/accelerate_configs`](https://github.com/huggingface/trl/tree/main/examples/accelerate_configs) directory. For example, to run training with ZeRO Stage 2, use the following command:

```bash
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml train.py
```

## Additional Resources

Consult the 🤗 Accelerate [documentation](https://huggingface.co/docs/accelerate/usage_guides/deepspeed) for more information about the DeepSpeed plugin.
