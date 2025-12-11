# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

from trl import AutoModelForCausalLMWithValueHead, create_reference_model

from .testing_utils import TrlTestCase


class TestReferenceModel(TrlTestCase):
    def setup_method(self):
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained("trl-internal-testing/tiny-GPT2LMHeadModel")
        self.test_input = torch.tensor([[0, 1, 2, 3]])
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1)
        self.layer_format = "pretrained_model.transformer.h.{layer}.attn.c_attn.weight"

    def test_independent_reference(self):
        layer_0 = self.layer_format.format(layer=0)
        layer_1 = self.layer_format.format(layer=1)

        ref_model = create_reference_model(self.model)

        first_layer_before = self.model.get_parameter(layer_0).data.clone()
        last_layer_before = self.model.get_parameter(layer_1).data.clone()  # the model only has 2 layers

        first_ref_layer_before = ref_model.get_parameter(layer_0).data.clone()
        last_ref_layer_before = ref_model.get_parameter(layer_1).data.clone()

        output = self.model(input_ids=self.test_input, labels=self.test_input)
        output[1].backward()
        self.optimizer.step()

        first_layer_after = self.model.get_parameter(layer_0).data.clone()
        last_layer_after = self.model.get_parameter(layer_1).data.clone()

        first_ref_layer_after = ref_model.get_parameter(layer_0).data.clone()
        last_ref_layer_after = ref_model.get_parameter(layer_1).data.clone()

        # before optimization ref and model are identical
        assert (first_layer_before == first_ref_layer_before).all()
        assert (last_layer_before == last_ref_layer_before).all()

        # ref model stays identical after optimization
        assert (first_ref_layer_before == first_ref_layer_after).all()
        assert (last_ref_layer_before == last_ref_layer_after).all()

        # optimized model changes
        assert not (first_layer_before == first_layer_after).all()
        assert not (last_layer_before == last_layer_after).all()

    def test_shared_layers(self):
        layer_0 = self.layer_format.format(layer=0)
        layer_1 = self.layer_format.format(layer=1)

        ref_model = create_reference_model(self.model, num_shared_layers=1)

        first_layer_before = self.model.get_parameter(layer_0).data.clone()
        second_layer_before = self.model.get_parameter(layer_1).data.clone()

        first_ref_layer_before = ref_model.get_parameter(layer_0).data.clone()
        second_ref_layer_before = ref_model.get_parameter(layer_1).data.clone()

        output = self.model(input_ids=self.test_input, labels=self.test_input)
        output[1].backward()
        self.optimizer.step()

        first_layer_after = self.model.get_parameter(layer_0).data.clone()
        second_layer_after = self.model.get_parameter(layer_1).data.clone()

        first_ref_layer_after = ref_model.get_parameter(layer_0).data.clone()
        second_ref_layer_after = ref_model.get_parameter(layer_1).data.clone()

        # before optimization ref and model are identical
        assert (first_layer_before == first_ref_layer_before).all()
        assert (second_layer_before == second_ref_layer_before).all()

        # ref model stays identical after optimization
        assert (first_ref_layer_before == first_ref_layer_after).all()
        assert (second_ref_layer_before == second_ref_layer_after).all()

        # first layer of optimized model stays the same
        assert (first_layer_before == first_layer_after).all()

        # other layers in optimized model change
        assert not (second_layer_before == second_layer_after).all()
