"""Async GRPO trainer with LoRA.  Run on GPU 1.

Usage::

    CUDA_VISIBLE_DEVICES=1 accelerate launch \\
        -m trl.experimental.async_grpo.lora.example.train
"""

from __future__ import annotations

import logging
from pathlib import Path

from datasets import load_dataset
from omegaconf import OmegaConf

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.rewards import accuracy_reward

logger = logging.getLogger(__name__)
_CONF = Path(__file__).resolve().parent / "config.yaml"


def run() -> None:
    cfg = OmegaConf.load(_CONF)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    dataset = load_dataset(cfg.dataset.name, split=cfg.dataset.split)
    if cfg.dataset.get("max_samples") and cfg.dataset.max_samples > 0:
        dataset = dataset.select(range(int(cfg.dataset.max_samples)))
    logger.info("Dataset size: %d", len(dataset))

    t = cfg.training
    vllm_url = f"{cfg.vllm.server_url}:{cfg.vllm.server_port}"

    config = AsyncGRPOConfig(
        output_dir=t.output_dir,
        max_steps=t.max_steps,
        per_device_train_batch_size=t.per_device_train_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        num_generations=t.num_generations,
        max_inflight_tasks=t.max_inflight_tasks,
        max_completion_length=t.max_completion_length,
        learning_rate=t.learning_rate,
        logging_steps=t.logging_steps,
        save_strategy="no",
        temperature=t.temperature,
        report_to=t.report_to,
        seed=t.seed,
        bf16=cfg.model.dtype == "bfloat16",
        vllm_server_base_url=vllm_url,
        vllm_server_timeout=cfg.vllm.server_timeout,
        chat_template_kwargs={"enable_thinking": False},
        warmup_ratio=t.warmup_ratio,
        lr_scheduler_type=t.lr_scheduler_type,
        # LoRA
        use_lora=True,
        lora_adapter_path=cfg.lora.adapter_dir,
        lora_name=cfg.lora.name,
    )

    trainer = AsyncGRPOTrainer(
        model=cfg.lora.adapter_dir,
        args=config,
        train_dataset=dataset,
        reward_funcs=accuracy_reward,
    )
    trainer.train()
    trainer.save_model("final_model")


if __name__ == "__main__":
    run()
