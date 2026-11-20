"""Bench copy of prime-rl tools/convert_dcp_to_bf16.py @ cbd3702 with one patch (tied lm_head, see NOTES.md #19).

Convert a DCP trainer checkpoint into bf16 HF-format weights offline.

The trainer saves only DCP checkpoints; every HF export goes through this
script. It mirrors the trainer's broadcast save path: build the model, DCP-load
the checkpoint's model state, gather rank-parallel, convert to HF format, and
write sharded safetensors plus config/tokenizer assets.

The model and tokenizer configs are read from the run's resolved config
(``<run>/configs/latest/resolved/trainer.json``, ``sft.json`` or ``pretrain.json``). LoRA checkpoints are not
supported — the script exports full fine-tunes only.

Usage (from the prime-rl repo; more ranks = faster gathers and writes, and
models too big for one GPU need enough ranks to shard across):
    uv run python tools/convert_dcp_to_bf16.py <run>/checkpoints/step_{n} [output_dir]
    uv run torchrun --nproc-per-node 8 tools/convert_dcp_to_bf16.py \
        <run>/checkpoints/step_{n} [output_dir]

Writes to ``<ckpt_dir>/weights`` by default.
"""

import argparse
import json
import os
import shutil
import socket
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
from prime_rl.configs.trainer import ModelConfig, MoERuntimeConfig, TokenizerConfig
from prime_rl.trainer.ckpt import AppState
from prime_rl.trainer.model import setup_model, setup_tokenizer
from prime_rl.trainer.parallel_dims import get_parallel_dims, resolve_ep
from prime_rl.trainer.utils import setup_torch_distributed
from prime_rl.trainer.vlm import setup_processor
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger, setup_logger
from prime_rl.utils.pathing import get_config_dir
from prime_rl.utils.weights import (
    convert_state_dict_to_hf,
    gather_weights_parallel,
    save_state_dict_parallel,
)
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load


# Training kernels and parallelism settings do not apply to an offline export.
# Resolve parallelism for the converter's world size and use portable kernels.
CONVERSION_OVERRIDES = {
    "compile": None,
    "ac": None,
    "dp_replicate": 1,
    "cp": 1,
    "ep": "auto",
    "moe": MoERuntimeConfig(),
    "attn": "flash_attention_2",
}

RUN_CONFIG_NAMES = ("trainer.json", "sft.json", "pretrain.json")


def resolve_dcp_dir(ckpt_dir: Path) -> Path:
    """The DCP checkpoint id for a step dir (``step_{n}``) or its ``trainer`` subdir."""
    if (ckpt_dir / "trainer" / ".metadata").exists():
        return ckpt_dir / "trainer"
    if (ckpt_dir / ".metadata").exists():
        return ckpt_dir
    raise FileNotFoundError(f"No DCP checkpoint found at {ckpt_dir} (expected {ckpt_dir}/trainer/.metadata)")


def resolve_run_configs(step_dir: Path) -> tuple[ModelConfig, TokenizerConfig]:
    """Model/tokenizer configs from the run's resolved config."""
    logger = get_logger()
    config_dir = get_config_dir(step_dir.parent.parent)
    for name in RUN_CONFIG_NAMES:
        path = config_dir / name
        if path.exists():
            logger.info(f"Reading model config from {path}")
            run_config = json.loads(path.read_text())
            break
    else:
        raise FileNotFoundError(
            f"No resolved run config ({' or '.join(RUN_CONFIG_NAMES)}) found under {config_dir} "
            "- the checkpoint must live in its run directory"
        )

    model = ModelConfig(**run_config["model"])
    tokenizer = TokenizerConfig(**run_config["tokenizer"])
    return model.model_copy(update=CONVERSION_OVERRIDES), tokenizer


def check_not_lora(model_config: ModelConfig, dcp_dir: Path) -> None:
    if model_config.lora is not None:
        raise ValueError("LoRA checkpoints are not supported - dcp_to_bf16 exports full fine-tunes only")
    metadata = FileSystemReader(dcp_dir).read_metadata()
    lora_keys = [k for k in metadata.state_dict_metadata if "lora_A" in k or "lora_B" in k or ".base_layer." in k]
    if lora_keys:
        raise ValueError(
            f"Checkpoint contains LoRA keys (e.g. {lora_keys[0]}) - dcp_to_bf16 exports full fine-tunes only"
        )


def setup_single_process_env() -> None:
    """Default the torchrun env vars so the script also runs under plain ``python``."""
    if "RANK" in os.environ:
        return
    with socket.socket() as sock:
        sock.bind(("localhost", 0))
        free_port = sock.getsockname()[1]
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(free_port)
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["LOCAL_WORLD_SIZE"] = "1"


def save_model_assets(model, model_config: ModelConfig, tokenizer_config: TokenizerConfig, output_dir: Path) -> None:
    """Save model config, generation config, processor and tokenizer next to the weights."""
    model.config.save_pretrained(output_dir)
    if getattr(model, "generation_config", None):
        # training sets use_cache=False which can conflict with cache_implementation —
        # save with use_cache=True without mutating the model's config
        gen_config = deepcopy(model.generation_config)
        gen_config.use_cache = True
        gen_config.save_pretrained(output_dir)
    # Processor first: it saves its own (unmodified) tokenizer, which the configured
    # tokenizer (pad token, custom chat template) must override.
    processor = setup_processor(model_config)
    if processor is not None:
        processor.save_pretrained(output_dir)
    if tokenizer_config.name is None:
        tokenizer_config = tokenizer_config.model_copy(update={"name": model_config.name})
    setup_tokenizer(tokenizer_config).save_pretrained(output_dir)
    # Local trust_remote_code models localize auto_map to plain module names, so the
    # custom code files must travel with the weights.
    source_dir = Path(model_config.name)
    if source_dir.is_dir():
        for path in source_dir.glob("*.py"):
            shutil.copyfile(path, output_dir / path.name)


def load_and_convert(ckpt_dir: Path):
    """DCP-load a checkpoint and gather it as a rank-partial HF-format bf16 state dict.

    Returns ``(model, model_config, tokenizer_config, state_dict, step_dir)``.
    """
    logger = setup_logger("info")

    dcp_dir = resolve_dcp_dir(ckpt_dir)
    step_dir = dcp_dir.parent
    model_config, tokenizer_config = resolve_run_configs(step_dir)
    check_not_lora(model_config, dcp_dir)

    setup_single_process_env()
    setup_torch_distributed()

    resolve_ep(model_config)
    parallel_dims = get_parallel_dims(model_config)
    model = setup_model(model_config, parallel_dims, loading_from_checkpoint_later=True)

    logger.info(f"Loading DCP checkpoint from {dcp_dir}")
    dcp_load(state_dict={"app": AppState(model, [], None, None)}, checkpoint_id=dcp_dir)

    logger.info("Gathering and converting weights")
    state_dict = gather_weights_parallel(model, dtype=torch.bfloat16)
    if getattr(model.config, "tie_word_embeddings", False):
        # Bench patch: prime-rl's Qwen3.5 sets _tied_weights_keys = None, so the upstream loop raised TypeError.
        # The 2B checkpoint ships no lm_head (tied to embed_tokens), so drop it from the export like upstream intends.
        for key in getattr(model, "_tied_weights_keys", None) or [
            k for k in state_dict if k.endswith("lm_head.weight")
        ]:
            state_dict.pop(key, None)
    state_dict = convert_state_dict_to_hf(model, state_dict)
    return model, model_config, tokenizer_config, state_dict, step_dir


def convert(ckpt_dir: Path, output_dir: Path | None = None) -> Path:
    logger = get_logger()
    model, model_config, tokenizer_config, state_dict, step_dir = load_and_convert(ckpt_dir)
    output_dir = output_dir if output_dir is not None else step_dir / "weights"

    logger.info(f"Writing HF weights to {output_dir}")
    save_state_dict_parallel(state_dict, output_dir)
    if get_world().is_master:
        save_model_assets(model, model_config, tokenizer_config, output_dir)
        logger.info(f"Done: {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "ckpt_dir", type=Path, help="the DCP checkpoint (<run>/checkpoints/step_{n} or .../step_{n}/trainer)"
    )
    parser.add_argument("output_dir", type=Path, nargs="?", default=None, help="default: <ckpt_dir>/weights")
    args = parser.parse_args()
    convert(args.ckpt_dir, args.output_dir)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
