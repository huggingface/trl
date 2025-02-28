# Distributing Training

<Tip warning={true}>

Section under construction. Feel free to contribute!

</Tip>

## Multi-GPU Training with TRL

Training with multiple GPUs in TRL is seamless, thanks to `accelerate`. You can switch from single-GPU to multi-GPU training with a simple command:

```bash
accelerate launch your_script.py
```

This automatically distributes the workload across all available GPUs.

Under the hood, `accelerate` creates one model per GPU. Each process:
- Processes its own batch of data
- Computes the loss and gradients for that batch
- Shares gradient updates across all GPUs

![](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/multi_gpu.png)


The effective batch size is calculated as:

$$
\text{Batch Size} = \text{per\_device\_train\_batch\_size} \times \text{num\_devices} \times \text{gradient\_accumulation\_steps}
$$


To maintain a consistent batch size when scaling to multiple GPUs, make sure to update `per_device_train_batch_size` and `gradient_accumulation_steps` accordingly.

Example, these configurations are equivalent, and should yield the same results:

| Number of GPUs | Per device batch size | Gradient accumulation steps | Comments |
| --- | --- | --- | --- |
| 1 | 32 | 1 | Possibly high memory usage, but faster training |
| 1 | 4 | 8 | Lower memory usage, slower training |
| 8 | 4 | 1 | Multi-GPU to get the best of both worlds |

## Multi-Nodes Training

Coming soon!