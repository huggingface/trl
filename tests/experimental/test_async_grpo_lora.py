"""LoRA async GRPO tests.

Unit tests (no GPU / single GPU) validate config, model loading, and parameter
freezing.  The full end-to-end LoRA training test with NCCL weight sync requires
2 GPUs and is best run via the example scripts::

    bash trl/experimental/async_grpo/lora/example/run_vllm.sh    # GPU 0
    bash trl/experimental/async_grpo/lora/example/run_trainer.sh  # GPU 1
"""

from __future__ import annotations

import json
import logging
import os
import tempfile

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


MODEL_ID = "Qwen/Qwen3-4B"
LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGETS = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def _create_lora_adapter(base_model: str, output_dir: str) -> str:
    """Create a minimal LoRA adapter and save it."""
    marker = os.path.join(output_dir, "adapter_config.json")
    if os.path.exists(marker):
        return output_dir

    log.info("Loading base model %s ...", base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    peft_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGETS,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(output_dir)
    log.info("Adapter saved to %s", output_dir)
    return output_dir


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
class TestLoRAModelLoading:
    """Validates that a LoRA adapter loads correctly and params freeze properly.

    Does NOT require vLLM or NCCL -- runs on a single GPU.
    """

    @pytest.fixture(scope="class")
    def adapter_dir(self, tmp_path_factory):
        d = str(tmp_path_factory.mktemp("lora_adapter"))
        _create_lora_adapter(MODEL_ID, d)
        return d

    def test_adapter_created(self, adapter_dir):
        """adapter_config.json and adapter weights exist."""
        assert os.path.exists(os.path.join(adapter_dir, "adapter_config.json"))
        assert os.path.exists(os.path.join(adapter_dir, "adapter_model.safetensors"))
        log.info("Adapter files verified at %s", adapter_dir)

    def test_adapter_config_values(self, adapter_dir):
        """adapter_config.json has the expected rank and alpha."""
        with open(os.path.join(adapter_dir, "adapter_config.json")) as f:
            cfg = json.load(f)
        assert cfg["r"] == LORA_RANK
        assert cfg["lora_alpha"] == LORA_ALPHA
        assert set(cfg["target_modules"]) == set(LORA_TARGETS)
        log.info("Adapter config: r=%d, alpha=%d, targets=%s", cfg["r"], cfg["lora_alpha"], cfg["target_modules"])

    def test_model_loads_with_lora_params(self, adapter_dir):
        """AutoModelForCausalLM.from_pretrained(adapter_dir) produces lora_ params."""
        model = AutoModelForCausalLM.from_pretrained(
            adapter_dir, dtype=torch.bfloat16, device_map="auto",
        )
        lora_params = [n for n, _ in model.named_parameters() if "lora_" in n]
        non_lora_params = [n for n, _ in model.named_parameters() if "lora_" not in n]

        log.info("LoRA params: %d, non-LoRA params: %d", len(lora_params), len(non_lora_params))
        assert len(lora_params) > 0, "No lora_ parameters found after loading adapter"
        assert len(non_lora_params) > 0, "Model has no base parameters"

    def test_freeze_non_lora(self, adapter_dir):
        """When we freeze non-LoRA params, only lora_ params have requires_grad=True."""
        model = AutoModelForCausalLM.from_pretrained(
            adapter_dir, dtype=torch.bfloat16, device_map="auto",
        )
        lora_count = 0
        frozen_count = 0
        for name, param in model.named_parameters():
            param.requires_grad = "lora_" in name
            if param.requires_grad:
                lora_count += 1
            else:
                frozen_count += 1

        log.info("Trainable (LoRA): %d, Frozen: %d", lora_count, frozen_count)
        assert lora_count > 0
        assert frozen_count > 0

        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        assert all("lora_" in n for n in trainable), (
            f"Non-LoRA param is trainable: {[n for n in trainable if 'lora_' not in n]}"
        )

    def test_async_grpo_config_lora_fields(self, adapter_dir):
        """AsyncGRPOConfig accepts lora fields and validates them."""
        from trl.experimental.async_grpo import AsyncGRPOConfig

        config = AsyncGRPOConfig(
            output_dir="/tmp/test",
            use_lora=True,
            lora_adapter_path=adapter_dir,
            lora_name="sft",
            report_to="none",
        )
        assert config.use_lora is True
        assert config.lora_adapter_path == adapter_dir
        assert config.lora_name == "sft"
        log.info("AsyncGRPOConfig LoRA fields validated")

    def test_async_grpo_config_rejects_missing_adapter_path(self):
        """use_lora=True without lora_adapter_path raises ValueError."""
        from trl.experimental.async_grpo import AsyncGRPOConfig

        with pytest.raises(ValueError, match="lora_adapter_path"):
            AsyncGRPOConfig(
                output_dir="/tmp/test",
                use_lora=True,
                report_to="none",
            )
        log.info("Missing adapter path correctly rejected")
