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

import pytest
from datasets import Dataset

from trl.experimental.sdzero import SDZeroConfig, SRTConfig, SRTTrainer

from ..testing_utils import TrlTestCase


class FakeSRTTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(
        self,
        conversation,
        tokenize,
        add_generation_prompt=False,
        continue_final_message=False,
        **kwargs,
    ):
        self.calls.append(
            {
                "conversation": conversation,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "continue_final_message": continue_final_message,
                "kwargs": kwargs,
            }
        )
        if add_generation_prompt:
            return [10]
        if continue_final_message:
            return [10, 20, 21]
        return [10, 20, 21, 30]


class TestSDZeroTrainer(TrlTestCase):
    def test_sdzero_config_defaults_keep_full_logit_reverse_kl(self):
        config = SDZeroConfig(output_dir=self.tmp_dir)

        assert config.teacher_model_kind == "base"
        assert config.teacher_update_rate == 1.0
        assert config.teacher_sync_steps == 512
        assert config.distillation_mode == "full_logits"
        assert config.distillation_alpha == 1.0
        assert config.distillation_topk is None
        assert config.distillation_is_clip is None
        assert config.num_generations == 1

    def test_sdzero_config_rejects_topk_without_topk_mode(self):
        with pytest.raises(ValueError, match="`distillation_topk` is only valid"):
            SDZeroConfig(output_dir=self.tmp_dir, distillation_topk=5)

    def test_expand_srt_dataset_builds_revision_and_generation_masks(self):
        dataset = Dataset.from_dict(
            {
                "problem": ["Solve 2+2."],
                "y_init": ["4"],
                "control_prompt": ["Let me verify."],
                "y_revised": ["The answer is 4."],
            }
        )
        tokenizer = FakeSRTTokenizer()
        args = SRTConfig(output_dir=self.tmp_dir, chat_template_kwargs={"enable_thinking": False})

        expanded = SRTTrainer._expand_srt_dataset(dataset, tokenizer, args)

        assert expanded["input_ids"] == [[10, 20, 21, 30], [10, 20, 21, 30]]
        assert expanded["completion_mask"] == [[0, 0, 0, 1], [0, 1, 1, 1]]
        assert all(call["kwargs"] == {"enable_thinking": False} for call in tokenizer.calls)
