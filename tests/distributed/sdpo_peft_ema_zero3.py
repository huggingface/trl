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
from accelerate import Accelerator
from deepspeed import zero
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, TrainerControl, TrainerState

from trl.experimental.sdpo.teacher_sync import PEFTAdapterEMACallback


def main():
    accelerator = Accelerator()
    assert accelerator.state.deepspeed_plugin.zero_stage == 3
    accelerator.state.deepspeed_plugin.deepspeed_config["zero_optimization"]["stage3_param_persistence_threshold"] = 0
    model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
    model = get_peft_model(
        model,
        LoraConfig(r=4, lora_alpha=8, target_modules="all-linear", task_type="CAUSAL_LM"),
    )
    callback = PEFTAdapterEMACallback(model=model, update_rate=0.5, sync_steps=1, accelerator=accelerator)
    callback._initialize_teacher_adapter()
    assert "teacher" in model.peft_config

    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-4)
    dataloader = DataLoader(TensorDataset(torch.tensor([[1, 2, 3, 4]])), batch_size=1)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    input_ids = next(iter(dataloader))[0]
    loss = model(input_ids=input_ids, labels=input_ids).loss
    accelerator.backward(loss)
    optimizer.step()

    unwrapped_model = accelerator.unwrap_model(model)
    student_parameters = [parameter for parameter in unwrapped_model.parameters() if parameter.requires_grad]
    assert student_parameters and any(parameter.numel() == 0 for parameter in student_parameters)
    student_state = callback._get_student_state_dict()
    assert student_state and all(value.numel() > 0 for value in student_state.values())
    expected = {key: value.detach().clone() * 0.5 for key, value in student_state.items()}
    assert any(torch.count_nonzero(value) > 0 for value in expected.values())
    callback.on_step_end(None, TrainerState(global_step=1), TrainerControl())

    teacher_parameters = [parameter for name, parameter in unwrapped_model.named_parameters() if ".teacher." in name]
    assert teacher_parameters
    with zero.GatheredParameters(teacher_parameters, modifier_rank=None):
        teacher_state = get_peft_model_state_dict(unwrapped_model, adapter_name="teacher")
        assert teacher_state.keys() == expected.keys()
        for key, value in expected.items():
            torch.testing.assert_close(teacher_state[key], value, rtol=0, atol=0)
    assert unwrapped_model.active_adapter == "default"
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
