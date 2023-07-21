# Getting started with Stable Diffusion finetuning with reinforcement learning

The machinery for finetuning of Stable Diffusion models with reinforcement learning makes heavy use of HuggingFace's `diffuser`
library. A reason for  stating this is that getting started requires a bit of familiarity with the `diffuser` library concepts, mainly two of them.
`Pipeline` and `Scheduler`. 
Right out of the box (`diffusers` library), there isn't a `Pipeline` nor a `Scheduler` instance that is suitable for finetuning with reinforcement learning. Some adjustments need to made. 

There is an interface that is provided by this library that allows you to easily create a `Pipeline` and a `Scheduler` that are suitable for finetuning with reinforcement learning. There is a default implementation of this interface that you can use out of the box with the limitation that DDIMScheduler is the only supported `Scheduler`. Assuming the default implementation is sufficient and/or to get thing moving, refer to the training example alongside this readme. 

For a more detailed look into the interface and the default scheduler and pipeline implement, go [here](https://github.com/lvwerra/trl/models/modelling_sd_base.py)

Also in addition, there is the expectation of providing a reward function and a prompt function. The reward function is used to evaluate the generated images  and the prompt function is used to generate the prompts that are used to generate the images.

# Setting up the image logging hook function

Expect the function to be given a list of lists of the form
```python
[[image, prompt, prompt_metadata, rewards], ...]

```
and `image`, `prompt`, `prompt_metadata`, `rewards` are batched.
The last list in the lists of lists represents the last sample batch. You are likely to want to log this one
While you are free to log however you want the use of `wandb` or `tensorboard` is recommended.

Example code for logging sampled images with `wandb` is given below.

```python
# for logging these images to wandb

def image_outputs_hook(image_data, global_step, accelerate_logger):
    # extract the last one
    result = {}
    images, prompts, _, rewards = image_data[-1]
    for i, image in enumerate(images):
        pil = Image.fromarray(
            (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        )
        pil = pil.resize((256, 256))
        result[f"{prompts[i]:.25} | {rewards[i]:.2f}"] = [pil]
    accelerate_logger.log_images(
        result,
        step=global_step,
    )

```

for an example using `tensorboard` see the training example

# Credits

This work is heavily influenced by the repo [here](https://github.com/kvablack/ddpo-pytorch) and the associated paper.