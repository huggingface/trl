# Training with Jobs

[![model badge](https://img.shields.io/badge/All_models-HF_Jobs-blue)](https://huggingface.co/models?other=hf_jobs,trl)

[Hugging Face Jobs](https://huggingface.co/docs/hub/jobs) runs your training on Hugging Face GPUs. You pick the hardware for each run and pay only for the seconds it runs. The trained model is pushed to the Hub at the end.

In this guide, you'll learn how to:

- Launch a TRL training script with one command
- Follow a run and see its loss curves
- Run your own script, with its hardware settings in the file
- Keep checkpoints and resume an interrupted run
- Train on several GPUs

For how Jobs works in general, including hardware and pricing, see the [Jobs documentation](https://huggingface.co/docs/hub/jobs).

## Requirements

- A Hugging Face account with a positive [credit balance](https://huggingface.co/settings/billing). Jobs is pay-as-you-go: you only pay for the seconds you use.
- Logged in to the Hugging Face Hub (`hf auth login`)

## A first run

This command fine-tunes [Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) on the [Capybara](https://huggingface.co/datasets/trl-lib/Capybara) chat dataset with the TRL SFT script, then pushes the model to your account:

<hfoptions id="script_type">
<hfoption id="bash">

```bash
hf jobs uv run \
    --flavor a10g-small \
    --timeout 30m \
    --secrets HF_TOKEN \
    -- \
    https://raw.githubusercontent.com/huggingface/trl/refs/heads/main/trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/Capybara \
    --max_steps 100 \
    --output_dir Qwen2-0.5B-SFT \
    --push_to_hub
```

</hfoption>
<hfoption id="python">

```python
from huggingface_hub import run_uv_job

run_uv_job(
    "https://raw.githubusercontent.com/huggingface/trl/refs/heads/main/trl/scripts/sft.py",
    flavor="a10g-small",
    timeout="30m",
    secrets={"HF_TOKEN": "hf_..."},
    script_args=[
        "--model_name_or_path", "Qwen/Qwen2-0.5B-Instruct",
        "--dataset_name", "trl-lib/Capybara",
        "--max_steps", "100",
        "--output_dir", "Qwen2-0.5B-SFT",
        "--push_to_hub",
    ],
)
```

</hfoption>
</hfoptions>

The run takes about six minutes. The options before `--` are for Jobs: the hardware, a time limit and the token used to push the model. The arguments after the script URL go to the script. `hf jobs uv run` runs a Python script with [uv](https://docs.astral.sh/uv/guides/scripts/). `sft.py` lists its dependencies in a header at the top of the file, so Jobs installs TRL before the script starts. The Job stops when the script exits, and billing stops with it.

`--max_steps 100` keeps this first run short. Remove it for the full run: three epochs, the script's default, take about 2 h 20 min on `a10g-small`, so raise `--timeout` to `3h`. Jobs stops a run when it reaches its timeout, which is 30 minutes by default. A larger GPU such as `a100-large` finishes sooner. See [Hardware](https://huggingface.co/docs/hub/jobs-pricing) for the flavors and their prices.

Models trained on Jobs get an `hf_jobs` tag, which lists them on the [models trained with TRL on Jobs](https://huggingface.co/models?other=hf_jobs,trl) page.

## Follow the run

`hf jobs uv run` streams the logs and holds your terminal until the run ends. Ctrl+C stops only the log stream: the Job keeps running. Add `--detach` to get the Job ID back straight away, then:

```bash
hf jobs logs -f <job_id>   # stream the logs
hf jobs ps                 # list your running Jobs
hf jobs inspect <job_id>   # status, and the error message if it failed
hf jobs cancel <job_id>    # stop it
```

See [Manage Jobs](https://huggingface.co/docs/hub/jobs-manage) for more.

For loss curves, TRL logs to [Trackio](trackio_integration). Pass `--report_to trackio` to the script and name a Space for the dashboard with `--env TRACKIO_SPACE_ID=<your-username>/trackio`. Trackio creates the Space, and a bucket for the metrics, if they do not exist. Both are public by default. Any other tracker that Transformers supports works too: pass its name to `--report_to`, such as `wandb`, and its API key as a secret, such as `--secrets WANDB_API_KEY`.

## Run your own script

Write your training code in a Python file (for example, `train.py`). List its dependencies at the top of the file, in a [script header](https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies), the same way the TRL scripts do. Jobs installs them before the script starts:

```python
# /// script
# dependencies = [
#     "trl",
#     "peft",
# ]
# ///

from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

dataset = load_dataset("trl-lib/Capybara", split="train")

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    train_dataset=dataset,
    peft_config=LoraConfig(),
    args=SFTConfig(output_dir="Qwen2.5-0.5B-SFT", max_steps=100),
)
trainer.train()
trainer.push_to_hub()
```

Then launch it with the [`hf jobs` CLI](https://huggingface.co/docs/huggingface_hub/guides/cli#hf-jobs) or the Python API:

<hfoptions id="script_type">
<hfoption id="bash">

```bash
hf jobs uv run \
    --flavor a10g-small \
    --timeout 30m \
    --secrets HF_TOKEN \
    train.py
```

</hfoption>
<hfoption id="python">

```python
from huggingface_hub import run_uv_job

run_uv_job(
    "train.py",
    flavor="a10g-small",
    timeout="30m",
    secrets={"HF_TOKEN": "hf_..."},
)
```

</hfoption>
</hfoptions>

For a script without a header, list the dependencies at launch instead with `--with`, once per package: `--with trl --with peft` (`dependencies=["trl", "peft"]` in Python). The script can also be a URL, such as a GitHub raw link, a Gist or a file in a public Hub repo.

The launch settings can also live in the script. A `[tool.hf-jobs]` table in the header sets the hardware, timeout and secrets:

```python
# /// script
# dependencies = [
#     "trl",
#     "peft",
# ]
#
# [tool.hf-jobs]
# flavor = "a10g-small"
# timeout = "30m"
# secrets = ["HF_TOKEN"]
# ///
```

`hf jobs uv run train.py` then needs no options, and an option you pass still overrides the script. The script carries everything it needs to run, so you can share it, or come back to it later, without remembering which GPU and timeout it needs. The table can also set `image`, `env`, `volumes` and other launch options. `hf jobs uv run --dry-run train.py` shows the resolved settings and marks the values that come from the script. The table is read by the `hf` CLI only: `run_uv_job()` ignores it. See [Define the launch config in the script](https://huggingface.co/docs/hub/jobs-configuration#define-the-launch-config-in-the-script).

## Run any TRL script

The same pattern runs every TRL trainer script (SFT, DPO, GRPO, reward modeling and others) and every `.py` example in the [Examples Index](example_overview#index). Each declares its dependencies in a script header. Notebook-only examples cannot be submitted this way. The script arguments are the same as when you run the script locally.

## Keep checkpoints

A Job's disk is deleted when the Job ends, including when it reaches its timeout or fails. For a long run, save checkpoints outside the Job so they survive it.

The simplest way is to push each checkpoint to the output model repo as it is saved:

<hfoptions id="script_type">
<hfoption id="bash">

```bash
hf jobs uv run \
    --flavor a10g-small \
    --timeout 3h \
    --secrets HF_TOKEN \
    -- \
    https://raw.githubusercontent.com/huggingface/trl/refs/heads/main/trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/Capybara \
    --output_dir Qwen2-0.5B-SFT \
    --push_to_hub \
    --save_steps 500 \
    --hub_strategy checkpoint
```

</hfoption>
<hfoption id="python">

```python
from huggingface_hub import run_uv_job

run_uv_job(
    "https://raw.githubusercontent.com/huggingface/trl/refs/heads/main/trl/scripts/sft.py",
    flavor="a10g-small",
    timeout="3h",
    secrets={"HF_TOKEN": "hf_..."},
    script_args=[
        "--model_name_or_path", "Qwen/Qwen2-0.5B-Instruct",
        "--dataset_name", "trl-lib/Capybara",
        "--output_dir", "Qwen2-0.5B-SFT",
        "--push_to_hub",
        "--save_steps", "500",
        "--hub_strategy", "checkpoint",
    ],
)
```

</hfoption>
</hfoptions>

With `--hub_strategy checkpoint`, the most recent checkpoint, including the optimizer state, is kept in a `last-checkpoint` folder of the repo. To continue an interrupted run, mount the repo into a new Job and resume from that folder. Add these options to the same command:

```bash
    --volume hf://<your-username>/Qwen2-0.5B-SFT:/previous \
    ...
    --resume_from_checkpoint /previous/last-checkpoint
```

To keep several checkpoints outside the model repo, write them to a [Storage Bucket](https://huggingface.co/docs/hub/storage-buckets) instead. Mount an existing bucket into the Job and point `--output_dir` at it:

<hfoptions id="script_type">
<hfoption id="bash">

```bash
hf jobs uv run \
    --flavor a10g-small \
    --timeout 3h \
    --secrets HF_TOKEN \
    --volume hf://buckets/<your-username>/checkpoints:/checkpoints \
    -- \
    https://raw.githubusercontent.com/huggingface/trl/refs/heads/main/trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/Capybara \
    --output_dir /checkpoints/Qwen2-0.5B-SFT \
    --save_steps 500
```

</hfoption>
<hfoption id="python">

```python
from huggingface_hub import Volume, run_uv_job

run_uv_job(
    "https://raw.githubusercontent.com/huggingface/trl/refs/heads/main/trl/scripts/sft.py",
    flavor="a10g-small",
    timeout="3h",
    secrets={"HF_TOKEN": "hf_..."},
    volumes=[Volume(type="bucket", source="<your-username>/checkpoints", mount_path="/checkpoints")],
    script_args=[
        "--model_name_or_path", "Qwen/Qwen2-0.5B-Instruct",
        "--dataset_name", "trl-lib/Capybara",
        "--output_dir", "/checkpoints/Qwen2-0.5B-SFT",
        "--save_steps", "500",
    ],
)
```

</hfoption>
</hfoptions>

To continue an interrupted run, launch the same command again with `--resume_from_checkpoint /checkpoints/Qwen2-0.5B-SFT/checkpoint-<step>`. See [Write to a bucket as you go](https://huggingface.co/docs/hub/jobs-training#after-it-ends) for creating a bucket and [Volumes](https://huggingface.co/docs/hub/jobs-configuration#volumes) for the mount options.

> [!TIP]
> Each checkpoint holds the model and optimizer state, several times the model size. Add `--save_total_limit 2` to keep only the latest ones. If you also pass `--push_to_hub`, set `--hub_model_id`, or the repo is named after the last part of the output path.

## Multiple GPUs

A flavor with several GPUs shortens a run. Run the TRL image with `hf jobs run` and start one process per GPU with `accelerate launch`. This works for the TRL scripts and for your own training code.

### Speed up a TRL script

The SFT run from the start of this guide, on four GPUs:

<hfoptions id="script_type">
<hfoption id="bash">

```bash
hf jobs run \
    --flavor l4x4 \
    --timeout 30m \
    --secrets HF_TOKEN \
    huggingface/trl \
    -- \
    accelerate launch --num_processes 4 -m trl.scripts.sft \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/Capybara \
    --max_steps 100 \
    --output_dir Qwen2-0.5B-SFT \
    --push_to_hub
```

</hfoption>
<hfoption id="python">

```python
from huggingface_hub import run_job

run_job(
    image="huggingface/trl",
    command=[
        "accelerate", "launch", "--num_processes", "4", "-m", "trl.scripts.sft",
        "--model_name_or_path", "Qwen/Qwen2-0.5B-Instruct",
        "--dataset_name", "trl-lib/Capybara",
        "--max_steps", "100",
        "--output_dir", "Qwen2-0.5B-SFT",
        "--push_to_hub",
    ],
    flavor="l4x4",
    timeout="30m",
    secrets={"HF_TOKEN": "hf_..."},
)
```

</hfoption>
</hfoptions>

`-m trl.scripts.sft` is the same SFT script as in the first run, from the TRL installed in the image. `accelerate launch` needs to start the script itself, which is why this form uses `hf jobs run` and the image rather than `hf jobs uv run`. Set `--num_processes` to the number of GPUs in the flavor, here four L4s. This run takes about seven minutes. Without `--max_steps`, the full three epochs take roughly an hour, so raise `--timeout` to `2h`.

### Your own script

You are not limited to the TRL scripts. Any training code that runs with `accelerate launch` locally runs the same way on Jobs, such as a GRPO script with your own reward function. Put it in a folder and mount the folder into the Job. Jobs uploads the folder to a private bucket and mounts it read-only:

<hfoptions id="script_type">
<hfoption id="bash">

```bash
hf jobs run \
    --flavor l4x4 \
    --timeout 30m \
    --secrets HF_TOKEN \
    --volume ./my-project:/code \
    huggingface/trl \
    -- \
    accelerate launch --num_processes 4 /code/train_grpo.py
```

</hfoption>
<hfoption id="python">

```python
from huggingface_hub import run_job, sync_job_volume

code = sync_job_volume("./my-project", "/code")
run_job(
    image="huggingface/trl",
    command=["accelerate", "launch", "--num_processes", "4", "/code/train_grpo.py"],
    flavor="l4x4",
    timeout="30m",
    secrets={"HF_TOKEN": "hf_..."},
    volumes=[code],
)
```

</hfoption>
</hfoptions>

The script runs with the TRL installed in the image, so it needs no script header. See [Local directories](https://huggingface.co/docs/hub/jobs-configuration#local-directories). The next section explains how `hf jobs run` uses the image.

## Docker Images

Jobs runs your script with `uv`, which installs the dependencies declared in its `# /// script` header into a fresh environment. The TRL your script imports therefore comes from that header, not from the image, and the examples above need no `--image` at all.

A Docker image with TRL preinstalled is available at [huggingface/trl](https://hub.docker.com/r/huggingface/trl). Passing it to `hf jobs uv run` gives the job the image's system layer, such as its CUDA toolchain, which matters for dependencies that compile against it:

<hfoptions id="script_type">
<hfoption id="bash">

```bash
hf jobs uv run \
    --flavor a100-large \
    --secrets HF_TOKEN \
    --image huggingface/trl \
    train.py
```

</hfoption>
<hfoption id="python">

```python
from huggingface_hub import run_uv_job

run_uv_job(
    "train.py",
    flavor="a100-large",
    secrets={"HF_TOKEN": "hf_..."},
    image="huggingface/trl",
)
```

</hfoption>
</hfoptions>

To run the TRL that is installed in the image, use `hf jobs run` instead. It runs a command in the image directly, with no script header to resolve, so the image's own TRL is what executes. The `--` separates the command from the Jobs options, which is needed whenever the command itself takes options:

<hfoptions id="script_type">
<hfoption id="bash">

```bash
hf jobs run \
    --flavor a100-large \
    --secrets HF_TOKEN \
    huggingface/trl \
    -- \
    trl sft --model_name_or_path Qwen/Qwen2-0.5B-Instruct --dataset_name trl-lib/Capybara --output_dir Qwen2-0.5B-SFT
```

</hfoption>
<hfoption id="python">

```python
from huggingface_hub import run_job

run_job(
    image="huggingface/trl",
    command=[
        "trl", "sft",
        "--model_name_or_path", "Qwen/Qwen2-0.5B-Instruct",
        "--dataset_name", "trl-lib/Capybara",
        "--output_dir", "Qwen2-0.5B-SFT",
    ],
    flavor="a100-large",
    secrets={"HF_TOKEN": "hf_..."},
)
```

</hfoption>
</hfoptions>

The image is published under three kinds of tag:

| Tag      | Contents                                                             |
| -------- | -------------------------------------------------------------------- |
| `X.Y.Z`  | The TRL release of that version, built when that version is released |
| `latest` | The most recent release                                              |
| `dev`    | The `main` branch, rebuilt on every merge                            |

Use `dev` to run the development version, which is useful for trying a fix that has landed on `main` but is not released yet:

<hfoptions id="script_type">
<hfoption id="bash">

```bash
hf jobs run \
    --flavor a100-large \
    --secrets HF_TOKEN \
    huggingface/trl:dev \
    -- \
    trl sft --model_name_or_path Qwen/Qwen2-0.5B-Instruct --dataset_name trl-lib/Capybara --output_dir Qwen2-0.5B-SFT
```

</hfoption>
<hfoption id="python">

```python
from huggingface_hub import run_job

run_job(
    image="huggingface/trl:dev",
    command=[
        "trl", "sft",
        "--model_name_or_path", "Qwen/Qwen2-0.5B-Instruct",
        "--dataset_name", "trl-lib/Capybara",
        "--output_dir", "Qwen2-0.5B-SFT",
    ],
    flavor="a100-large",
    secrets={"HF_TOKEN": "hf_..."},
)
```

</hfoption>
</hfoptions>

Tags only select what is installed in the image, so they matter for `hf jobs run`. With `hf jobs uv run` the version comes from the script header instead, and the tag makes no difference to which TRL is imported.

Combining the two, so that `uv` resolves the script header while some imports still come from the image, needs extra flags. See [Reuse the image's packages and add dependencies with UV](https://huggingface.co/docs/hub/jobs-images#reuse-the-images-packages-and-add-dependencies-with-uv) for that form and the paths it requires.

Jobs runs on a Docker image from Hugging Face Spaces or Docker Hub, so you can also specify any custom image:

<hfoptions id="script_type">
<hfoption id="bash">

```bash
hf jobs uv run \
    --flavor a100-large \
    --secrets HF_TOKEN \
    --image <docker-image> \
    train.py
```

</hfoption>
<hfoption id="python">

```python
from huggingface_hub import run_uv_job

run_uv_job(
    "train.py",
    flavor="a100-large",
    secrets={"HF_TOKEN": "hf_..."},
    image="<docker-image>",
)
```

</hfoption>
</hfoptions>

> [!NOTE]
> [TRL Jobs](https://github.com/huggingface/trl-jobs) is a small wrapper that launches the TRL scripts on Jobs with preset configurations for some models, for example `trl-jobs sft --model_name Qwen/Qwen3-0.6B --dataset_name trl-lib/Capybara`.
