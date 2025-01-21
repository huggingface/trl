# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import gc
import sys
import tempfile
import unittest

import torch
from parameterized import parameterized
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, GenerationConfig

from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, create_reference_model


ALL_CAUSAL_LM_MODELS = [
    "trl-internal-testing/tiny-BloomForCausalLM",
    "trl-internal-testing/tiny-CohereForCausalLM",
    "trl-internal-testing/tiny-DbrxForCausalLM",
    "trl-internal-testing/tiny-FalconMambaForCausalLM",
    "trl-internal-testing/tiny-Gemma2ForCausalLM",
    "trl-internal-testing/tiny-GemmaForCausalLM",
    "trl-internal-testing/tiny-GPT2LMHeadModel",
    "trl-internal-testing/tiny-GPTNeoXForCausalLM",
    "trl-internal-testing/tiny-LlamaForCausalLM-3.1",
    "trl-internal-testing/tiny-LlamaForCausalLM-3.2",
    "trl-internal-testing/tiny-LlamaForCausalLM-3",
    "trl-internal-testing/tiny-MistralForCausalLM-0.1",
    "trl-internal-testing/tiny-MistralForCausalLM-0.2",
    "trl-internal-testing/tiny-OPTForCausalLM",
    "trl-internal-testing/tiny-Phi3ForCausalLM",
    "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
]

ALL_SEQ2SEQ_MODELS = [
    "trl-internal-testing/tiny-T5ForConditionalGeneration",
    "trl-internal-testing/tiny-BartModel",
]


class BaseTester:
    class VHeadModelTester(unittest.TestCase):
        all_model_names = None
        trl_model_class = None
        transformers_model_class = None

        def test_value_head(self):
            r"""
            Test if the v-head is added to the model successfully
            """
            for model_name in self.all_model_names:
                model = self.trl_model_class.from_pretrained(model_name)
                self.assertTrue(hasattr(model, "v_head"))

        def test_value_head_shape(self):
            r"""
            Test if the v-head has the correct shape
            """
            for model_name in self.all_model_names:
                model = self.trl_model_class.from_pretrained(model_name)
                self.assertEqual(model.v_head.summary.weight.shape[0], 1)

        def test_value_head_init_random(self):
            r"""
            Test if the v-head has been randomly initialized.
            We can check that by making sure the bias is different
            than zeros by default.
            """
            for model_name in self.all_model_names:
                model = self.trl_model_class.from_pretrained(model_name)
                self.assertFalse(
                    torch.allclose(model.v_head.summary.bias, torch.zeros_like(model.v_head.summary.bias))
                )

        def test_value_head_not_str(self):
            r"""
            Test if the v-head is added to the model successfully, by passing a non `PretrainedModel`
            as an argument to `from_pretrained`.
            """
            for model_name in self.all_model_names:
                pretrained_model = self.transformers_model_class.from_pretrained(model_name)
                model = self.trl_model_class.from_pretrained(pretrained_model)
                self.assertTrue(hasattr(model, "v_head"))

        @unittest.skipIf(sys.platform.startswith("win"), "Skipping on Windows")
        def test_from_save_trl(self):
            """
            Test if the model can be saved and loaded from a directory and get the same weights
            Including the additional modules (e.g. v_head)
            """
            for model_name in self.all_model_names:
                model = self.trl_model_class.from_pretrained(model_name)

                with tempfile.TemporaryDirectory() as tmp_dir:
                    model.save_pretrained(tmp_dir)

                    model_from_save = self.trl_model_class.from_pretrained(tmp_dir)

                # Check if the weights are the same
                for key in model_from_save.state_dict():
                    self.assertTrue(torch.allclose(model_from_save.state_dict()[key], model.state_dict()[key]))

        @unittest.skipIf(sys.platform.startswith("win"), "Skipping on Windows")
        def test_from_save_trl_sharded(self):
            """
            Test if the model can be saved and loaded from a directory and get the same weights - sharded case
            """
            for model_name in self.all_model_names:
                model = self.trl_model_class.from_pretrained(model_name)

                with tempfile.TemporaryDirectory() as tmp_dir:
                    model.save_pretrained(tmp_dir)

                    model_from_save = self.trl_model_class.from_pretrained(tmp_dir)

                # Check if the weights are the same
                for key in model_from_save.state_dict():
                    self.assertTrue(torch.allclose(model_from_save.state_dict()[key], model.state_dict()[key]))

        @unittest.skipIf(sys.platform.startswith("win"), "Skipping on Windows")
        def test_from_save_transformers_sharded(self):
            """
            Test if the model can be saved and loaded using transformers and get the same weights - sharded case
            """
            for model_name in self.all_model_names:
                transformers_model = self.trl_model_class.transformers_parent_class.from_pretrained(model_name)

                trl_model = self.trl_model_class.from_pretrained(model_name)

                with tempfile.TemporaryDirectory() as tmp_dir:
                    trl_model.save_pretrained(tmp_dir, max_shard_size="1MB")
                    transformers_model_from_save = self.trl_model_class.transformers_parent_class.from_pretrained(
                        tmp_dir
                    )

                # Check if the weights are the same
                for key in transformers_model.state_dict():
                    self.assertTrue(
                        torch.allclose(
                            transformers_model_from_save.state_dict()[key], transformers_model.state_dict()[key]
                        )
                    )

        @unittest.skipIf(sys.platform.startswith("win"), "Skipping on Windows")
        def test_from_save_transformers(self):
            """
            Test if the model can be saved and loaded using transformers and get the same weights.
            We override the test of the super class to check if the weights are the same.
            """
            for model_name in self.all_model_names:
                transformers_model = self.trl_model_class.transformers_parent_class.from_pretrained(model_name)

                trl_model = self.trl_model_class.from_pretrained(model_name)

                with tempfile.TemporaryDirectory() as tmp_dir:
                    trl_model.save_pretrained(tmp_dir)
                    transformers_model_from_save = self.trl_model_class.transformers_parent_class.from_pretrained(
                        tmp_dir
                    )

                # Check if the weights are the same
                for key in transformers_model.state_dict():
                    self.assertTrue(
                        torch.allclose(
                            transformers_model_from_save.state_dict()[key], transformers_model.state_dict()[key]
                        )
                    )

                # Check if the trl model has the same keys as the transformers model
                # except the v_head
                for key in trl_model.state_dict():
                    if "v_head" not in key:
                        self.assertIn(key, transformers_model.state_dict())
                        # check if the weights are the same
                        self.assertTrue(
                            torch.allclose(trl_model.state_dict()[key], transformers_model.state_dict()[key])
                        )

                # check if they have the same modules
                self.assertEqual(
                    set(transformers_model_from_save.state_dict().keys()),
                    set(transformers_model.state_dict().keys()),
                )


class CausalLMValueHeadModelTester(BaseTester.VHeadModelTester, unittest.TestCase):
    """
    Testing suite for v-head models.
    """

    all_model_names = ALL_CAUSAL_LM_MODELS
    trl_model_class = AutoModelForCausalLMWithValueHead
    transformers_model_class = AutoModelForCausalLM

    def tearDown(self):
        # free memory
        gc.collect()

    def test_inference(self):
        r"""
        Test if the model can be used for inference and outputs 3 values
        - logits, loss, and value states
        """
        EXPECTED_OUTPUT_SIZE = 3

        for model_name in self.all_model_names:
            model = self.trl_model_class.from_pretrained(model_name)
            input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
            outputs = model(input_ids)

            # Check if the outputs are of the right size - here
            # we always output 3 values - logits, loss, and value states
            self.assertEqual(len(outputs), EXPECTED_OUTPUT_SIZE)

    def test_dropout_config(self):
        r"""
        Test if we instantiate a model by adding `summary_drop_prob` to the config
        it will be added to the v_head
        """
        for model_name in self.all_model_names:
            pretrained_model = self.transformers_model_class.from_pretrained(model_name)
            pretrained_model.config.summary_dropout_prob = 0.5
            model = self.trl_model_class.from_pretrained(pretrained_model)

            # Check if v head of the model has the same dropout as the config
            self.assertEqual(model.v_head.dropout.p, pretrained_model.config.summary_dropout_prob)

    def test_dropout_kwargs(self):
        r"""
        Test if we instantiate a model by adding `summary_drop_prob` to the config
        it will be added to the v_head
        """
        for model_name in self.all_model_names:
            v_head_kwargs = {"summary_dropout_prob": 0.5}

            model = self.trl_model_class.from_pretrained(model_name, **v_head_kwargs)

            # Check if v head of the model has the same dropout as the config
            self.assertEqual(model.v_head.dropout.p, 0.5)

            model = self.trl_model_class.from_pretrained(model_name, summary_dropout_prob=0.5)

            # Check if v head of the model has the same dropout as the config
            self.assertEqual(model.v_head.dropout.p, 0.5)

    @parameterized.expand(ALL_CAUSAL_LM_MODELS)
    def test_generate(self, model_name):
        r"""
        Test if `generate` works for every model
        """
        generation_config = GenerationConfig(max_new_tokens=9)
        model = self.trl_model_class.from_pretrained(model_name)
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])

        # Just check if the generation works
        _ = model.generate(input_ids, generation_config=generation_config)

    def test_transformers_bf16_kwargs(self):
        r"""
        Test if the transformers kwargs are correctly passed
        Here we check that loading a model in half precision works as expected, i.e. the weights of
        the `pretrained_model` attribute is loaded in half precision and you can run a dummy
        forward pass without any issue.
        """
        for model_name in self.all_model_names:
            trl_model = self.trl_model_class.from_pretrained(model_name, torch_dtype=torch.bfloat16)

            lm_head_namings = ["lm_head", "embed_out", "output_layer"]

            self.assertTrue(
                any(hasattr(trl_model.pretrained_model, lm_head_naming) for lm_head_naming in lm_head_namings),
                "Can't test the model because it doesn't have any of the expected lm_head namings",
            )

            for lm_head_naming in lm_head_namings:
                if hasattr(trl_model.pretrained_model, lm_head_naming):
                    self.assertEqual(getattr(trl_model.pretrained_model, lm_head_naming).weight.dtype, torch.bfloat16)

            dummy_input = torch.LongTensor([[0, 1, 0, 1]])

            # check dummy forward pass works in half precision
            _ = trl_model(dummy_input)

    @unittest.skip("This test needs to be run manually due to HF token issue.")
    def test_push_to_hub(self):
        for model_name in self.all_model_names:
            model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
            if "sharded" in model_name:
                model.push_to_hub(model_name + "-ppo", use_auth_token=True, max_shard_size="1MB")
            else:
                model.push_to_hub(model_name + "-ppo", use_auth_token=True)

            model_from_pretrained = AutoModelForCausalLMWithValueHead.from_pretrained(model_name + "-ppo")
            # check all keys
            self.assertEqual(model.state_dict().keys(), model_from_pretrained.state_dict().keys())

            for name, param in model.state_dict().items():
                self.assertTrue(
                    torch.allclose(param, model_from_pretrained.state_dict()[name]),
                    f"Parameter {name} is not the same after push_to_hub and from_pretrained",
                )


class Seq2SeqValueHeadModelTester(BaseTester.VHeadModelTester, unittest.TestCase):
    """
    Testing suite for v-head models.
    """

    all_model_names = ALL_SEQ2SEQ_MODELS
    trl_model_class = AutoModelForSeq2SeqLMWithValueHead
    transformers_model_class = AutoModelForSeq2SeqLM

    def tearDown(self):
        # free memory
        gc.collect()

    def test_inference(self):
        r"""
        Test if the model can be used for inference and outputs 3 values
        - logits, loss, and value states
        """
        EXPECTED_OUTPUT_SIZE = 3

        for model_name in self.all_model_names:
            model = self.trl_model_class.from_pretrained(model_name)
            input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
            decoder_input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
            outputs = model(input_ids, decoder_input_ids=decoder_input_ids)

            # Check if the outputs are of the right size - here
            # we always output 3 values - logits, loss, and value states
            self.assertEqual(len(outputs), EXPECTED_OUTPUT_SIZE)

    def test_dropout_config(self):
        r"""
        Test if we instantiate a model by adding `summary_drop_prob` to the config
        it will be added to the v_head
        """
        for model_name in self.all_model_names:
            pretrained_model = self.transformers_model_class.from_pretrained(model_name)
            pretrained_model.config.summary_dropout_prob = 0.5
            model = self.trl_model_class.from_pretrained(pretrained_model)

            # Check if v head of the model has the same dropout as the config
            self.assertEqual(model.v_head.dropout.p, pretrained_model.config.summary_dropout_prob)

    def test_dropout_kwargs(self):
        r"""
        Test if we instantiate a model by adding `summary_drop_prob` to the config
        it will be added to the v_head
        """
        for model_name in self.all_model_names:
            v_head_kwargs = {"summary_dropout_prob": 0.5}

            model = self.trl_model_class.from_pretrained(model_name, **v_head_kwargs)

            # Check if v head of the model has the same dropout as the config
            self.assertEqual(model.v_head.dropout.p, 0.5)

            model = self.trl_model_class.from_pretrained(model_name, summary_dropout_prob=0.5)

            # Check if v head of the model has the same dropout as the config
            self.assertEqual(model.v_head.dropout.p, 0.5)

    @parameterized.expand(ALL_SEQ2SEQ_MODELS)
    def test_generate(self, model_name):
        r"""
        Test if `generate` works for every model
        """
        generation_config = GenerationConfig(max_new_tokens=9)
        model = self.trl_model_class.from_pretrained(model_name)
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        decoder_input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])

        # Just check if the generation works
        _ = model.generate(input_ids, decoder_input_ids=decoder_input_ids, generation_config=generation_config)

    def test_raise_error_not_causallm(self):
        # Test with a model without a LM head
        model_id = "trl-internal-testing/tiny-T5ForConditionalGeneration"
        # This should raise a ValueError
        with self.assertRaises(ValueError):
            pretrained_model = AutoModel.from_pretrained(model_id)
            _ = self.trl_model_class.from_pretrained(pretrained_model)

    @unittest.skip("This test needs to be run manually due to HF token issue.")
    def test_push_to_hub(self):
        for model_name in self.all_model_names:
            model = self.trl_model_class.from_pretrained(model_name)
            if "sharded" in model_name:
                model.push_to_hub(model_name + "-ppo", use_auth_token=True, max_shard_size="1MB")
            else:
                model.push_to_hub(model_name + "-ppo", use_auth_token=True)

            model_from_pretrained = self.trl_model_class.from_pretrained(model_name + "-ppo")
            # check all keys
            self.assertEqual(model.state_dict().keys(), model_from_pretrained.state_dict().keys())

            for name, param in model.state_dict().items():
                self.assertTrue(
                    torch.allclose(param, model_from_pretrained.state_dict()[name]),
                    f"Parameter {name} is not the same after push_to_hub and from_pretrained",
                )

    def test_transformers_bf16_kwargs(self):
        r"""
        Test if the transformers kwargs are correctly passed
        Here we check that loading a model in half precision works as expected, i.e. the weights of
        the `pretrained_model` attribute is loaded in half precision and you can run a dummy
        forward pass without any issue.
        """
        for model_name in self.all_model_names:
            trl_model = self.trl_model_class.from_pretrained(model_name, torch_dtype=torch.bfloat16)

            lm_head_namings = self.trl_model_class.lm_head_namings

            self.assertTrue(
                any(hasattr(trl_model.pretrained_model, lm_head_naming) for lm_head_naming in lm_head_namings)
            )

            for lm_head_naming in lm_head_namings:
                if hasattr(trl_model.pretrained_model, lm_head_naming):
                    self.assertTrue(getattr(trl_model.pretrained_model, lm_head_naming).weight.dtype == torch.bfloat16)

            dummy_input = torch.LongTensor([[0, 1, 0, 1]])

            # check dummy forward pass works in half precision
            _ = trl_model(input_ids=dummy_input, decoder_input_ids=dummy_input)


class ReferenceModelTest(unittest.TestCase):
    def setUp(self):
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
        self.assertTrue((first_layer_before == first_ref_layer_before).all())
        self.assertTrue((last_layer_before == last_ref_layer_before).all())

        # ref model stays identical after optimization
        self.assertTrue((first_ref_layer_before == first_ref_layer_after).all())
        self.assertTrue((last_ref_layer_before == last_ref_layer_after).all())

        # optimized model changes
        self.assertFalse((first_layer_before == first_layer_after).all())
        self.assertFalse((last_layer_before == last_layer_after).all())

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
        self.assertTrue((first_layer_before == first_ref_layer_before).all())
        self.assertTrue((second_layer_before == second_ref_layer_before).all())

        # ref model stays identical after optimization
        self.assertTrue((first_ref_layer_before == first_ref_layer_after).all())
        self.assertTrue((second_ref_layer_before == second_ref_layer_after).all())

        # first layer of optimized model stays the same
        self.assertTrue((first_layer_before == first_layer_after).all())

        # other layers in optimized model change
        self.assertFalse((second_layer_before == second_layer_after).all())
