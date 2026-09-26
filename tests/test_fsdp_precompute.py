# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.utils import is_peft_available

from trl import DPOTrainer, KTOTrainer

from .testing_utils import require_peft


if is_peft_available():
    from peft import LoraConfig, get_peft_model


@require_peft
@pytest.mark.parametrize("trainer_class", [DPOTrainer, KTOTrainer])
def test_fsdp1_precompute_configures_peft_before_preparation(tmp_path, trainer_class):
    trainer = object.__new__(trainer_class)
    config = LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2
    )
    trainer.model = get_peft_model(
        LlamaForCausalLM(config).to(torch.bfloat16), LoraConfig(r=2, target_modules=["q_proj", "v_proj"])
    )
    parameter_dtypes = {name: param.dtype for name, param in trainer.model.named_parameters()}
    assert set(parameter_dtypes.values()) == {torch.bfloat16, torch.float32}
    trainer.ref_model = None
    trainer.is_fsdp_enabled = True
    trainer.is_deepspeed_enabled = False
    trainer._precompute_engine = None
    trainer._precompute_model_hash = "fsdp1-peft-preparation"
    trainer.calculate_KL = False
    trainer.args = SimpleNamespace(dataloader_num_workers=0, dataloader_pin_memory=False)
    trainer.data_collator = lambda rows: torch.tensor([row["x"] for row in rows])
    plugin = SimpleNamespace(fsdp_version=1, auto_wrap_policy=None)
    events = []

    def prepare(obj):
        if isinstance(obj, DataLoader):
            return obj
        assert obj is trainer.model
        # Exercise the real PEFT policy, not a stubbed update helper. Without
        # this setup, FSDP1 flattens FP32 trainable LoRA and BF16 base weights together.
        assert plugin.auto_wrap_policy is not None
        lora_layer = trainer.model.base_model.model.model.layers[0].self_attn.q_proj.lora_A["default"]
        assert plugin.auto_wrap_policy(lora_layer, recurse=False, nonwrapped_numel=lora_layer.weight.numel())
        events.append("prepare")
        return obj

    trainer.accelerator = SimpleNamespace(
        state=SimpleNamespace(fsdp_plugin=plugin),
        prepare=prepare,
        gather_for_metrics=lambda value: value,
        is_main_process=True,
        wait_for_everyone=lambda: None,
    )

    def reference_forward(model, batch):
        assert events[0] == "prepare"
        assert model is trainer.model_wrapped is trainer._precompute_engine
        assert not model.training
        events.append("forward")
        return torch.zeros(len(batch)), torch.zeros(len(batch)) if trainer_class is DPOTrainer else None

    trainer.compute_ref_log_probs = reference_forward
    for name in ("train", "eval"):
        path = tmp_path / name
        Dataset.from_dict({"x": [1, 2]}).save_to_disk(path)
        result = trainer._precompute_ref_logps(load_from_disk(path), name, 1)
        column = "ref_chosen_logps" if trainer_class is DPOTrainer else "ref_logps"
        assert list(result[column]) == [0.0, 0.0]
    assert events == ["prepare"] + ["forward"] * 4
    assert {name: param.dtype for name, param in trainer.model.named_parameters()} == parameter_dtypes
