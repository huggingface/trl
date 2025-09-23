# Training with Jobs

[![](https://img.shields.io/badge/All_models-HF_Jobs-blue)](https://huggingface.co/models?other=hf_jobs,trl)

[Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) lets you run training scripts on fully managed infrastructure—no need to manage GPUs or local environment setup.

In this guide, you'll learn how to:

* Use [TRL Jobs](https://github.com/huggingface/trl-jobs) to easily run pre-optimized TRL training
* Run any TRL training script with uv scripts

For general details about Hugging Face Jobs (hardware selection, job monitoring, etc.), see the [Jobs documentation](https://huggingface.co/docs/huggingface_hub/guides/jobs).

## Requirements

* A [Pro](https://hf.co/pro), [Team](https://hf.co/enterprise), or [Enterprise](https://hf.co/enterprise) plan
* Logged in to the Hugging Face Hub (`hf auth login`)

## Using TRL Jobs

[TRL Jobs](https://github.com/huggingface/trl-jobs) is a high-level wrapper around Hugging Face Jobs and TRL that streamlines training. It provides optimized default configurations so you can start quickly without manually tuning parameters.

Example:

```bash
pip install trl-jobs
trl-jobs sft --model_name Qwen/Qwen3-0.6B --dataset_name trl-lib/Capybara
```

TRL Jobs supports everything covered in this guide, with additional optimizations to simplify workflows.

## Using uv Scripts

For more control, you can run Hugging Face Jobs directly with your own scripts, using [uv scripts](https://docs.astral.sh/uv/guides/scripts/).

Create a Python script (e.g., `train.py`) containing your training code:

```python
from datasets import load_dataset
from trl import SFTTrainer

dataset = load_dataset("trl-lib/Capybara", split="train")
trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    train_dataset=dataset,
)
trainer.train()
trainer.push_to_hub("Qwen2.5-0.5B-SFT")
```

Launch the job using either the [`hf jobs` CLI](https://huggingface.co/docs/huggingface_hub/guides/cli#hf-jobs) or the Python API:

<hfoptions id="script_type">
<hfoption id="bash">

```bash
hf jobs uv run \
    --flavor a100-large \
    --with trl \
    --secrets HF_TOKEN \
    train.py
```

</hfoption>
<hfoption id="python">

```python
from huggingface_hub import run_uv_job

run_uv_job(
    "train.py",
    dependencies=["trl"],
    flavor="a100-large",
    secrets={"HF_TOKEN": "hf_..."},
)
```

</hfoption>
</hfoptions>

To run successfully, the script needs:

* **TRL installed**: Use the `--with trl` flag or the `dependencies` argument. uv installs these dependencies automatically before running the script.
* **An authentication token**: Required to push the trained model (or perform other authenticated operations). Provide it with the `--secrets HF_TOKEN` flag or the `secrets` argument.

<Tip warning={true}>

When training with Jobs, be sure to:

* **Set a sufficient timeout**. Jobs time out after 30 minutes by default. If your job exceeds the timeout, it will fail and all progress will be lost. See [Setting a custom timeout](https://huggingface.co/docs/huggingface_hub/guides/jobs#setting-a-custom-timeout).
* **Push the model to the Hub**. The Jobs environment is ephemeral—files are deleted when the job ends. If you don’t push the model, it will be lost.

</Tip>

You can also run a script directly from a URL:

<hfoptions id="script_type">
<hfoption id="bash">

```bash
hf jobs uv run \
    --flavor a100-large \
    --with trl \
    --secrets HF_TOKEN \
    "https://gist.githubusercontent.com/qgallouedec/eb6a7d20bd7d56f9c440c3c8c56d2307/raw/69fd78a179e19af115e4a54a1cdedd2a6c237f2f/train.py"
```

</hfoption>
<hfoption id="python">

```python
from huggingface_hub import run_uv_job

run_uv_job(
    "https://gist.githubusercontent.com/qgallouedec/eb6a7d20bd7d56f9c440c3c8c56d2307/raw/69fd78a179e19af115e4a54a1cdedd2a6c237f2f/train.py",
    flavor="a100-large",
    dependencies=["trl"],
    secrets={"HF_TOKEN": "hf_..."},
)
```

</hfoption>
</hfoptions>

To make a script self-contained, declare dependencies at the top:

```python
# /// script
# dependencies = [
#     "trl",
#     "peft",
# ]
# ///

from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer

dataset = load_dataset("trl-lib/Capybara", split="train")

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    train_dataset=dataset,
    peft_config=LoraConfig(),
)
trainer.train()
trainer.push_to_hub("Qwen2.5-0.5B-SFT")
```

You can then run the script without specifying dependencies:

<hfoptions id="script_type">
<hfoption id="bash">

```bash
hf jobs uv run \
    --flavor a100-large \
    --secrets HF_TOKEN \
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
)
```

</hfoption>
</hfoptions>

<Tip>

TRL example scripts are fully uv-compatible, so you can run a complete training workflow directly on Jobs. You can customize training with standard script arguments plus hardware and secrets:

<hfoptions id="script_type">
<hfoption id="bash">

```bash
hf jobs uv run \
    --flavor a100-large \
    --secrets HF_TOKEN \
    https://raw.githubusercontent.com/huggingface/trl/refs/heads/main/examples/scripts/prm.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/prm800k \
    --output_dir Qwen2-0.5B-Reward \
    --push_to_hub
```

</hfoption>
<hfoption id="python">

```python
from huggingface_hub import run_uv_job

run_uv_job(
    "https://raw.githubusercontent.com/huggingface/trl/refs/heads/main/examples/scripts/prm.py",
    flavor="a100-large",
    secrets={"HF_TOKEN": "hf_..."},
    script_args=[
        "--model_name_or_path", "Qwen/Qwen2-0.5B-Instruct",
        "--dataset_name", "trl-lib/prm800k",
        "--output_dir", "Qwen2-0.5B-Reward",
        "--push_to_hub"
    ]
)
```

</hfoption>
</hfoptions>

See the full list of examples in [Maintained examples](example_overview#maintained-examples).

</Tip>

### Docker Images

An up-to-date Docker image with all TRL dependencies is available at [huggingface/trl](https://hub.docker.com/r/huggingface/trl) and can be used directly with Hugging Face Jobs:

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

Jobs runs on a Docker image from Hugging Face Spaces or Docker Hub, so you can also specify any custom image:

<hfoptions id="script_type">
<hfoption id="bash">

```bash
hf jobs uv run \
    --flavor a100-large \
    --secrets HF_TOKEN \
    --image <docker-image> \
    --secrets HF_TOKEN \
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
