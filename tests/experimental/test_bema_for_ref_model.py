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

import torch
from datasets import load_dataset
from transformers import TrainerControl, TrainerState
from transformers.utils import is_peft_available

from trl import DPOConfig
from trl.experimental.bema_for_ref_model import BEMACallback, DPOTrainer

from ..testing_utils import TrlTestCase, require_peft


if is_peft_available():
    from peft import LoraConfig


class TestBEMACallback(TrlTestCase):
    def setup_method(self):
        self.dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

    def test_update_ref_model(self):
        """Test that BEMACallback updates a separate ref_model with BEMA weights."""
        training_args = DPOConfig(output_dir=self.tmp_dir, report_to="none")
        bema_callback = BEMACallback(update_freq=2, update_ref_model=True, ref_model_update_freq=2)
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=self.dataset,
            callbacks=[bema_callback],
        )
        trainer.train()

    @require_peft
    def test_update_ref_model_peft(self):
        training_args = DPOConfig(output_dir=self.tmp_dir, report_to="none", use_cpu=True, bf16=False)
        bema_callback = BEMACallback(update_freq=100, update_ref_model=True, ref_model_update_freq=1)
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=self.dataset,
            peft_config=LoraConfig(),
            callbacks=[bema_callback],
        )
        assert trainer.ref_model is None

        state = TrainerState(global_step=1)
        control = TrainerControl()
        bema_callback.on_train_begin(training_args, state, control, model=trainer.model)

        base_model = trainer.model.get_base_model()
        parameters = dict(base_model.named_parameters())
        adapter_name = next(name for name, param in parameters.items() if param.requires_grad)
        base_name = next(name for name, param in parameters.items() if not param.requires_grad)
        adapter_before = parameters[adapter_name].detach().clone()

        running_state = bema_callback.running_model.state_dict()
        with torch.no_grad():
            running_state[adapter_name].fill_(7)
            running_state[base_name].fill_(3)

        bema_callback.on_step_end(training_args, state, control, model=trainer.model, ref_model=trainer.ref_model)

        torch.testing.assert_close(parameters[adapter_name], adapter_before)
        torch.testing.assert_close(parameters[base_name], torch.full_like(parameters[base_name], 3))
