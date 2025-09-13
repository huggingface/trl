# Training using Jobs

[![](https://img.shields.io/badge/All_models-HF_Jobs-blue)](https://huggingface.co/models?other=hf_jobs,trl)

[Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) lets you run training scripts on fully managed infrastructure (no need to handle GPUs, dependencies, or environment setup locally). This makes it easy to scale and monitor your experiments directly from the Hub.

In this guide, you’ll learn how to:

- Run TRL training scripts using Jobs.
- Configure hardware, timeouts, environment variables, and secrets.
- Monitor and manage training jobs efficiently.
- Leverage [`trl-jobs`](https://github.com/huggingface/trl-jobs) to simplify and optimize your TRL workflows.

## Requirements

- [Pro](https://hf.co/pro), [Team](https://hf.co/enterprise), or [Enterprise](https://hf.co/enterprise) plan.
- Logged into the Hugging Face Hub (`hf auth login`).

## Using TRL Jobs

[TRL Jobs](https://github.com/huggingface/trl-jobs) is a high-level wrapper around HF Jobs and TRL that simplifies the training process. It comes with optimized default configurations for training models, making it easier to get started without manually specifying all the parameters.

To run a training job with TRL Jobs, simply use:

```bash
pip install trl-jobs
trl-jobs sft --model_name Qwen/Qwen3-0.6B --dataset_name trl-lib/Capybara
```

TRL Jobs supports the same functionality covered in this guide, but with additional optimizations to streamline your workflows.

## Using uv Scripts

If you need more control over your training process, you can use Hugging Face Jobs directly with your own scripts. A convenient way to do this is with [uv scripts](https://docs.astral.sh/uv/guides/scripts/).

You can launch Jobs using either the [`hf jobs` CLI](https://huggingface.co/docs/huggingface_hub/guides/cli#hf-jobs) or the Python API:

<hfoptions id="script_type">
<hfoption id="bash">

```bash
hf jobs uv run --flavor a100-large train.py
```

</hfoption>
<hfoption id="python">

```python
from huggingface_hub import run_uv_job

run_uv_job("train.py", flavor="a100-large")
```

</hfoption>
</hfoptions>

<Tip warning={true}>

Depending on your workflow, your script may need to access private datasets or upload a trained model to the Hub. Such operations require authentication, so make sure to include a valid authentication token.

<hfoptions id="script_type">
<hfoption id="bash">

```bash
hf jobs uv run --flavor a100-large --secrets HF_TOKEN=hf_... train.py
```

</hfoption>
<hfoption id="python">

```python
from huggingface_hub import run_uv_job

run_uv_job(
    "train.py",
    flavor="a100-large",
    secrets={"HF_TOKEN": "hf_..."}
)
```

</hfoption>
</hfoptions>

</Tip>

You can also run scripts directly from a URL and even pass script arguments.

<hfoptions id="script_type">
<hfoption id="bash">

```bash
hf jobs uv run --flavor a100-large "https://raw.githubusercontent.com/huggingface/trl/main/trl/scripts/sft.py" --model_name_or_path Qwen/Qwen2-0.5B --dataset_name trl-lib/Capybara
```

</hfoption>
<hfoption id="python">

```python
from huggingface_hub import run_uv_job
run_uv_job(
    "trl/scripts/sft.py",
    flavor="a100-large",
    script_args=[
        "--model_name_or_path", "Qwen/Qwen2-0.5B",
        "--dataset_name", "trl-lib/Capybara",
    ]
)
```

</hfoption>
</hfoptions>

Since it runs using a Docker Image from Hugging Face Spaces or Docker Hub, you can also specify it:

<hfoptions id="script_type">
<hfoption id="bash">

```bash
hf jobs uv run --flavor a100-large --image <docker-image> train.py
```

</hfoption>
<hfoption id="python">

```python
from huggingface_hub import run_uv_job
run_uv_job(
    "train.py",
    flavor="a100-large",
    image="<docker-image>",
)
```

</hfoption>
</hfoptions>

We provide an up-to-date Docker image with all the dependencies required by TRL ([huggingface/trl](https://hub.docker.com/r/huggingface/trl)), which you can use directly in HF Jobs.

<hfoptions id="script_type">
<hfoption id="bash">

```bash
hf jobs uv run --flavor a100-large --image huggingface/trl train.py
```

</hfoption>
<hfoption id="python">

```python
from huggingface_hub import run_uv_job
run_uv_job(
    "train.py",
    flavor="a100-large",
    image="huggingface/trl",
)
```

</hfoption>
</hfoptions>



## Rest ---
You can also run jobs without UV:

<hfoptions id="script_type">
<hfoption id="bash">

In this case, we give the cli the Docker image and run it as:

```bash
hf jobs run --flavor a100-large pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel python -c "import torch; print(torch.cuda.get_device_name())"
```

</hfoption>
<hfoption id="python">

```python
from huggingface_hub import run_job
run_job(
    image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
    command=["python", "-c", "import torch; print(torch.cuda.get_device_name())"],
    flavor="a100-large",
)
```

</hfoption>
</hfoptions>

### Adding Dependencies with UV

You can also provide dependencies directly in the `uv run` command:

<hfoptions id="script_type">
<hfoption id="bash">

Using the `--with` flag.

```bash
hf jobs uv run \
    --flavor a100-large \
    --secrets HF_TOKEN \
    --with transformers \
    --with torch \
    "https://raw.githubusercontent.com/huggingface/trl/main/trl/scripts/sft.py" \
    --model_name_or_path Qwen/Qwen2-0.5B  \
    --dataset_name trl-lib/Capybara
```

</hfoption>
<hfoption id="python">

Using the `dependencies` argument.

```python
from huggingface_hub import run_uv_job
run_uv_job(
    "https://raw.githubusercontent.com/huggingface/trl/main/trl/scripts/sft.py",
    dependencies=["transformers", "torch"]
    token="hf...",
    flavor="a100-large",
    script_args=[
        "--model_name_or_path", "Qwen/Qwen2-0.5B",
        "--dataset_name", "trl-lib/Capybara",
    ]
)
```

</hfoption>
</hfoptions>

### Hardware and Timeout Settings

Jobs allow you to select a specific hardware configuration using the `--flavor` flag. As of 08/25, the available options are:

**CPU:** `cpu-basic`, `cpu-upgrade`  
**GPU:** `t4-small`, `t4-medium`, `l4x1`, `l4x4`, `a10g-small`, `a10g-large`, `a10g-largex2`, `a10g-largex4`, `a100-large`  
**TPU:** `v5e-1x1`, `v5e-2x2`, `v5e-2x4`  

You can always check the latest list of supported hardware flavors in [Spaces config reference](https://huggingface.co/docs/hub/en/spaces-config-reference).

By default, jobs have a **30-minute timeout**, after which they will automatically stop. For long-running tasks like training, you can increase the timeout as needed. Supported time units are:

- `s`: seconds
- `m`: minutes
- `h`: hours
- `d`: days

Example with a 2-hour timeout:

<hfoptions id="script_type">
<hfoption id="bash">

Using the `--timeout` flag:

```bash
hf jobs uv run \
    --timeout 2h \
    --flavor a100-large \
    --secrets HF_TOKEN \
    --with transformers \
    --with torch \
    "https://raw.githubusercontent.com/huggingface/trl/main/trl/scripts/sft.py" \
    --model_name_or_path Qwen/Qwen2-0.5B  \
    --dataset_name trl-lib/Capybara
```

</hfoption>
<hfoption id="python">

Using the `timeout` argument:

```python
from huggingface_hub import run_uv_job
run_uv_job(
    "https://raw.githubusercontent.com/huggingface/trl/main/trl/scripts/sft.py",
    timeout="2h",
    token="hf...",
    flavor="a100-large",
    script_args=[
        "--model_name_or_path", "Qwen/Qwen2-0.5B",
        "--dataset_name", "trl-lib/Capybara",
    ]
)
```

</hfoption>
</hfoptions>







### 

To make your training scripts compatible with uv scripts, ensure they include a `dependencies` list at the top. Dependencies are specified at the top of the script using this structure:

```python
# /// script
# dependencies = [
#     "trl",
#     "peft",
# ]
# ///

from trl import SFTTrainer
from datasets import load_dataset
from peft import LoraConfig

dataset = load_dataset("trl-lib/Capybara", split="train")

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    train_dataset=dataset,
    peft_config=LoraConfig(),
)
trainer.train()
```

When you run the UV script, these dependencies are automatically installed. In the above example, `trl` and `peft` would be installed before the script runs.

<Tip>

All examples and scripts in TRL are compatible with `uv`, allowing seamless execution with Jobs. You can check the full list of examples in [Maintained examples](example_overview#maintained-examples).

</Tip>







### Environment Variables, Secrets, and Token

You can pass environment variables, secrets, and your auth token to your jobs. 

<hfoptions id="script_type">
<hfoption id="bash">

Using the `--env`, `--secrets`, and/or `--token` options.

```bash
hf jobs uv run \
    trl/scripts/sft.py \
    --flavor a100-large \
    --env FOO=foo \
    --env BAR=bar \
    --secrets HF_TOKEN=HF_TOKEN \
    --secrets MY_SECRET=password \
    --token hf...
```

</hfoption>
<hfoption id="python">


Using the `env`, `secrets`, and/or `token` arguments.

```python
from huggingface_hub import run_uv_job
run_uv_job(
    "trl/scripts/sft.py",
    env={"FOO": "foo", "BAR": "bar"},
    secrets={"MY_SECRET": "psswrd"},
    token="hf..."
)
```

</hfoption>
</hfoptions>

## Training and Evaluating a Model with Jobs

TRL example scripts are fully UV-compatible, allowing you to run a complete training workflow directly on Jobs. You can customize the training by providing the usual script arguments, along with hardware specifications and secrets.  

To evaluate your training runs, in addition to reviewing the job logs, you can use [**Trackio**](https://huggingface.co/blog/trackio), a lightweight experiment tracking library. Trackio enables end-to-end experiment management on the Hugging Face Hub. All TRL example scripts already support reporting to Trackio via the `report_to` argument. Using this feature saves your experiments in an interactive HF Space, making it easy to monitor metrics, compare runs, and track progress over time.

<hfoptions id="script_type">
<hfoption id="bash">

```bash
hf jobs uv run \
    --flavor a100-large \
    --secrets HF_TOKEN \
    "trl/scripts/sft.py" \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --eos_token '<|im_end|>' \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir Qwen2-0.5B-SFT \
    --report_to trackio \
    --push_to_hub
```

</hfoption>
<hfoption id="python">

```python
from huggingface_hub import run_uv_job

run_uv_job(
    "trl/scripts/sft.py",
    flavor="a100-large",
    secrets={"HF_TOKEN": "your_hf_token"},
    script_args=[
        "--model_name_or_path", "Qwen/Qwen2-0.5B",
        "--dataset_name", "trl-lib/Capybara",
        "--learning_rate", "2.0e-5",
        "--num_train_epochs", "1",
        "--packing",
        "--per_device_train_batch_size", "2",
        "--gradient_accumulation_steps", "8",
        "--eos_token", "<|im_end|>",
        "--eval_strategy", "steps",
        "--eval_steps", "100",
        "--output_dir", "Qwen2-0.5B-SFT",
        "--report_to", "trackio",
        "--push_to_hub"
    ]
)
```

</hfoption>
</hfoptions>

## Monitoring and Managing Jobs

After launching a job, you can track its progress on the [Jobs page](https://huggingface.co/settings/jobs). Additionally, Jobs provides CLI and Python commands to check status, view logs, or cancel a job.

<hfoptions id="script_type">
<hfoption id="bash">

```bash
# List your jobs
hf jobs ps -a

# List your running jobs
hf jobs ps 

# Inspect the status of a job
hf jobs inspect

# View logs from a job
hf jobs logs job_id

# Cancel a job
hf jobs cancel job_id
```

</hfoption>
<hfoption id="python">

```python
from huggingface_hub import list_jobs, inspect_job, fetch_job_logs, cancel_job

# List your jobs
jobs = list_jobs()
jobs[0]

# List your running jobs
running_jobs = [job for job in list_jobs() if job.status.stage == "RUNNING"]

# Inspect the status of a job
inspect_job(job_id=job_id)

# View logs from a job
for log in fetch_job_logs(job_id=job_id):
    print(log)

# Cancel a job
cancel_job(job_id=job_id)
```

</hfoption>
</hfoptions>

## Best Practices and Tips

- Choose hardware that fits the size of your model and dataset for optimal performance.
- Training jobs can be long-running. Consider increasing the default timeout.
- Reuse training and evaluation scripts whenever possible to streamline workflows.
