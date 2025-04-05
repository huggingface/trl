# Installation
You can install TRL either from PyPI or from source:

## PyPI
Install the library with pip or [uv](https://docs.astral.sh/uv/):

<hfoptions id="install">
<hfoption id="uv">

uv is a fast Rust-based Python package and project manager. Refer to [Installation](https://docs.astral.sh/uv/getting-started/installation/) for installation instructions), .

```bash
uv pip install trl
```

</hfoption>
<hfoption id="pip">

```bash
pip install trl
```

</hfoption>
</hfoptions>

## Source
You can also install the latest version from source. First clone the repo and then run the installation with `pip`:

```bash
git clone https://github.com/huggingface/trl.git
cd trl/
pip install -e .
```

If you want the development install you can replace the pip install with the following:

```bash
pip install -e ".[dev]"
```
