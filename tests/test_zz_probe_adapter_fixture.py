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

"""Throwaway probe. Not for merge.

Checks whether regenerating the peft tiny models with `init_lora_weights=False` would give
`TestUseAdapter` something real to assert, before the PR is rewritten around that assumption.
"""

import torch
from transformers import AutoModelForCausalLM, set_seed
from transformers.utils import is_peft_available

from .testing_utils import TrlTestCase, require_peft


if is_peft_available():
    from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model


BASE = "trl-internal-testing/tiny-Qwen3ForCausalLM"
INPUT_IDS = torch.tensor([[1, 2, 3], [4, 5, 6]])


@require_peft
class TestProbeAdapterFixture(TrlTestCase):
    def _build_and_save(self, seed, subdir):
        """Exactly what peft_qwen3_for_causal_lm.py would do with init_lora_weights=False."""
        set_seed(seed)
        model = AutoModelForCausalLM.from_pretrained(BASE)
        model = get_peft_model(model, LoraConfig(init_lora_weights=False))
        path = f"{self.tmp_dir}/{subdir}"
        model.save_pretrained(path)
        return path

    def test_1_current_published_fixture_is_identity(self):
        """Negative control: today's fixture cannot distinguish enabled from disabled."""
        model = AutoPeftModelForCausalLM.from_pretrained(
            "trl-internal-testing/tiny-PeftModel", adapter_name="my_adapter"
        )
        enabled = model(INPUT_IDS).logits
        with model.disable_adapter():
            disabled = model(INPUT_IDS).logits
        assert torch.equal(enabled, disabled), "fixture is NOT identity, the premise of the PR is wrong"

    def test_2_regenerated_fixture_changes_logits(self):
        """The adapter must actually do something once saved and reloaded from disk."""
        path = self._build_and_save(42, "a")
        model = AutoPeftModelForCausalLM.from_pretrained(path, adapter_name="my_adapter")
        enabled = model(INPUT_IDS).logits
        with model.disable_adapter():
            disabled = model(INPUT_IDS).logits
        assert not torch.equal(enabled, disabled)

    def test_3_two_seeds_give_different_adapters(self):
        """The second repo exists to differ from the first, so the seeds must separate them."""
        path_a = self._build_and_save(42, "a")
        path_b = self._build_and_save(43, "b")

        model = AutoPeftModelForCausalLM.from_pretrained(path_a, adapter_name="my_adapter_1")
        model.load_adapter(path_b, "my_adapter_2")

        model.set_adapter("my_adapter_1")
        logits_1 = model(INPUT_IDS).logits
        model.set_adapter("my_adapter_2")
        logits_2 = model(INPUT_IDS).logits
        assert not torch.equal(logits_1, logits_2)

    def test_4_same_seed_reproduces_the_adapter(self):
        """Seeding has to be what separates them, not luck."""
        path_a = self._build_and_save(42, "a")
        path_b = self._build_and_save(42, "b")

        model = AutoPeftModelForCausalLM.from_pretrained(path_a, adapter_name="my_adapter_1")
        model.load_adapter(path_b, "my_adapter_2")

        model.set_adapter("my_adapter_1")
        logits_1 = model(INPUT_IDS).logits
        model.set_adapter("my_adapter_2")
        logits_2 = model(INPUT_IDS).logits
        assert torch.equal(logits_1, logits_2)
