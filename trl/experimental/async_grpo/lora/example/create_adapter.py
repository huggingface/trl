"""Create a LoRA adapter from a base model.  Run once before training.

Usage::

python -m trl.experimental.async_grpo.lora.example.create_adapter
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)
_CONF = Path(__file__).resolve().parent / "config.yaml"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = OmegaConf.load(_CONF)

    output_dir = Path(cfg.lora.adapter_dir)
    if (output_dir / "adapter_config.json").exists():
        log.info("Adapter already exists at %s — skipping creation", output_dir)
        return

    log.info("Loading base model %s …", cfg.model.name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    peft_config = LoraConfig(
        r=cfg.lora.rank,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg.lora.target_modules),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)

    AutoTokenizer.from_pretrained(cfg.model.name).save_pretrained(output_dir)
    log.info("Adapter saved to %s", output_dir)


if __name__ == "__main__":
    main()
