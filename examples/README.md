# Examples

Each example lives in its own self-contained folder. The full index and layout conventions are documented at https://huggingface.co/docs/trl/example_overview.

## Adding a new example

- **Name the folder** after the method and what makes the example distinctive: the task (`grpo_wordle`), the model (`sft_gpt_oss`), or the technique (`grpo_qlora`). Load folder-local assets relative to the script (`Path(__file__).parent / ...`), not the working directory.
- **An example tells a story.** A bare single-trainer training script is not an example: stable trainers already have one as a CLI command ([`trl/scripts`](../trl/scripts)), and every trainer's documentation page has a runnable snippet.
- **Scripts declare their dependencies** in a `# /// script` header and **document the exact run command(s)** in the module docstring (`python examples/<folder>/<script>.py ...`), including multi-GPU variants where relevant.
- **Add a row to the Index table** in [`docs/source/example_overview.md`](../docs/source/example_overview.md), with a Colab badge if the example has a notebook that runs on free Colab.
