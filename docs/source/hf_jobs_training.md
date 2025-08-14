# Training a model with TRL using HF Jobs

[HF Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) lets you run training scripts on fully managed infrastructure (no need to handle GPUs, dependencies, or environment setup locally). This makes it easy to scale and monitor your experiments directly from the Hub.

In this guide, you’ll learn how to:
- Run TRL training scripts using HF Jobs.
- Configure hardware, timeouts, environment variables, and secrets.
- Monitor and manage jobs from the CLI or Python.

**Requirements**
- Pro, Team, or Enterprise plan.
- Logged into the Hugging Face Hub (`hf auth login`).

## Preparing your Script

You can launch HF Jobs using either the [`hf jobs` CLI](https://huggingface.co/docs/huggingface_hub/guides/cli#hf-jobs) or the Python API. A convenient option is to use [UV scripts](https://docs.astral.sh/uv/guides/scripts/), which package all dependencies directly into a single Python file. You can run them like this:

<hfoptions id="script_type">
<hfoption id="bash">

```bash
hf jobs uv run --flavor a10g-small "https://raw.githubusercontent.com/huggingface/trl/main/trl/scripts/sft.py" 
```

The script can also be a local file:

```bash
hf jobs uv run --flavor a10g-small sft.py
```

Since it runs using a Docker Image from Hugging Face Spaces or Docker Hub, you can also specify it:

```bash
hf jobs uv run --flavor a10g-small --image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel sft.py
```

</hfoption>
<hfoption id="python">

```python
from huggingface_hub import run_uv_job
run_uv_job(
    "https://raw.githubusercontent.com/huggingface/trl/main/trl/scripts/sft.py",
    flavor="a10g-small"
)
```

The script can also be a local file:

```python
from huggingface_hub import run_uv_job
run_uv_job(
    "sft.py",
    flavor="a10g-small"
)
```

Since it runs using a Docker Image from Hugging Face Spaces or Docker Hub, you can also specify it:

```python
from huggingface_hub import run_uv_job
run_uv_job(
    "sft.py",
    flavor="a10g-small",
    image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
)
```


</hfoption>
</hfoptions>

You can also run jobs without uv:

<hfoptions id="script_type">
<hfoption id="bash">

In this case, we give the command the Docker image and command to run:

```bash
hf jobs run --flavor a10g-small pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel python -c "import torch; print(torch.cuda.get_device_name())"
```

</hfoption>
<hfoption id="python">

```python
from huggingface_hub import run_job
run_job(
    image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
    command=["python", "-c", "import torch; print(torch.cuda.get_device_name())"],
    flavor="a10g-small",
)
```

</hfoption>
</hfoptions>


### Adding Dependencies with UV

All example scripts in TRL are compatible with `uv`, allowing seamless execution with HF Jobs. You can check the full list of examples [here](example_overview#maintained-examples).  

Dependencies are specified at the top of the script using this structure:

```python
# /// script
# dependencies = [
#     "trl @ git+https://github.com/huggingface/trl.git",
#     "peft",
# ]
# ///
```

When you run the UV script, these dependencies are automatically installed. In the example above, `trl` and `peft` would be installed before the script runs.

You can also provide dependencies directly in the run command:

<hfoptions id="script_type">
<hfoption id="bash">

```bash
hf jobs uv run \
    --flavor a10g-small \
    --dependencies transformers torch \
    "https://raw.githubusercontent.com/huggingface/trl/main/trl/scripts/sft.py" 
```

</hfoption>
<hfoption id="python">

```python
from huggingface_hub import run_uv_job
run_uv_job(
    "https://raw.githubusercontent.com/huggingface/trl/main/trl/scripts/sft.py",
    dependencies=["transformers", "torch"]
)
```

</hfoption>
</hfoptions>

### Hardware and Timeout Settings

HF Jobs allows you to select a specific hardware configuration using the `--flavor` flag. As of 08/25, the available options are:

**CPU:** `cpu-basic`, `cpu-upgrade`  
**GPU:** `t4-small`, `t4-medium`, `l4x1`, `l4x4`, `a10g-small`, `a10g-large`, `a10g-largex2`, `a10g-largex4`, `a100-large`  
**TPU:** `v5e-1x1`, `v5e-2x2`, `v5e-2x4`  

You can always check the latest list of supported hardware flavors [here](https://huggingface.co/docs/hub/en/spaces-config-reference).

Example usage:

<hfoptions id="script_type">
<hfoption id="bash">

```bash
hf jobs uv run \
    --flavor a10g-small \
    "https://raw.githubusercontent.com/huggingface/trl/main/trl/scripts/sft.py" 
```

</hfoption>
<hfoption id="python">

```python
from huggingface_hub import run_uv_job
run_uv_job(
    "https://raw.githubusercontent.com/huggingface/trl/main/trl/scripts/sft.py",
    flavor="a10g-small"
)
```

</hfoption>
</hfoptions>

By default, jobs have a **30-minute timeout**, after which they will automatically stop. For long-running tasks like training, you can increase the timeout using the `--timeout` flag. Supported time units are:

- `s`: seconds
- `m`: minutes
- `h`: hours
- `d`: days

Example with a 2-hour timeout:

<hfoptions id="script_type">
<hfoption id="bash">

```bash
hf jobs uv run \
    --timeout 2h \
    --flavor a10g-small \
    "https://raw.githubusercontent.com/huggingface/trl/main/trl/scripts/sft.py" 
```

</hfoption>
<hfoption id="python">

```python
from huggingface_hub import run_uv_job
run_uv_job(
    "https://raw.githubusercontent.com/huggingface/trl/main/trl/scripts/sft.py",
    timeout="2h"
)
```

</hfoption>
</hfoptions>

### Environment Variables, Secrets and Token

You can pass environment variables, secrets, and your auth token to your jobs using the `--env`, `--secrets`, and/or `--token` options.

<hfoptions id="script_type">
<hfoption id="bash">

```bash
hf jobs uv run \             
    --flavor a100-large \
    --env FOO=foo \
    --env BAR=bar \
    --secrets HF_TOKEN \
    --secrets MY_SECRET=password \
    --token hf...
```

</hfoption>
<hfoption id="python">

```python
from huggingface_hub import run_uv_job
run_uv_job(
    "https://raw.githubusercontent.com/huggingface/trl/main/trl/scripts/sft.py",
    env={"FOO": "foo", "BAR": "bar"},
    secrets={"MY_SECRET": "psswrd"},
    token="hf..."
)
```

</hfoption>
</hfoptions>


## Training a Model with HF Jobs

TRL example scripts are fully UV-compatible, so you can run a training procedure directly with HF Jobs. You can customize the training by passing the usual script arguments along with hardware and secrets options.

<hfoptions id="script_type">
<hfoption id="bash">

```bash
hf jobs uv run \             
    --flavor a100-large \
    --secrets HF_TOKEN \
    "https://raw.githubusercontent.com/huggingface/trl/main/trl/scripts/sft.py" \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eos_token '<|im_end|>' \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir Qwen2-0.5B-SFT \
    --push_to_hub
```

</hfoption>
<hfoption id="python">

```python
from huggingface_hub import run_uv_job

run_uv_job(
    "https://raw.githubusercontent.com/huggingface/trl/main/trl/scripts/sft.py",
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
        "--gradient_checkpointing",
        "--eos_token", "<|im_end|>",
        "--eval_strategy", "steps",
        "--eval_steps", "100",
        "--output_dir", "Qwen2-0.5B-SFT",
        "--push_to_hub"
    ]
)
```

</hfoption>
</hfoptions>

## Monitoring and Managing Jobs

After launching a job, you can track its progress on the [Jobs page](https://huggingface.co/settings/jobs). Additionally, HF Jobs provides CLI and Python commands to check status, view logs, or cancel a job.

**Checking status**

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

```python
from huggingface_hub import list_jobs

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


## Best Practices and Tips

- Choose hardware that fits the size of your model and dataset for optimal performance.
- Training jobs can be long-running—consider increasing the default timeout.
- Reuse training and evaluation scripts whenever possible to streamline workflows.
