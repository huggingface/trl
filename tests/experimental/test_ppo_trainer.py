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

import gc
import os

import pytest
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
)
from transformers.utils import is_peft_available

from trl.experimental.ppo import (
    AutoModelForCausalLMWithValueHead,
    AutoModelForSeq2SeqLMWithValueHead,
    PPOConfig,
    PPOTrainer,
)
from trl.experimental.ppo.ppo_trainer import batch_generation, masked_mean, masked_var, masked_whiten

from ..testing_utils import (
    TrlTestCase,
    require_bitsandbytes,
    require_peft,
    require_torch_gpu_if_bnb_not_multi_backend_enabled,
)


if is_peft_available():
    from peft import LoraConfig, get_peft_model


ALL_CAUSAL_LM_MODELS = [
    "trl-internal-testing/tiny-BloomForCausalLM",
    "trl-internal-testing/tiny-CohereForCausalLM",
    # "trl-internal-testing/tiny-FalconMambaForCausalLM",  # FalconMambaForCausalLM modeling seems to be broken for now
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


class TestBatchGeneration(TrlTestCase):
    def setup_method(self):
        # Initialize the tokenizer
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        self.generation_config = GenerationConfig(
            max_new_tokens=128,
            temperature=0.5,
            do_sample=True,
            top_k=0,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Example input
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")
        self.examples = dataset["messages"]
        self.mini_batch_size = 3

    def test_mini_batch_generation(self):
        batch = [
            self.tokenizer.apply_chat_template(example[:-1], add_generation_prompt=True, tokenize=False)
            for example in self.examples
        ]
        queries = self.tokenizer(batch, padding=True, return_tensors="pt")["input_ids"].to(self.device)
        bs, context_length = queries.shape

        query_responses, logits = batch_generation(
            self.model, queries, self.mini_batch_size, self.tokenizer.pad_token_id, self.generation_config
        )

        max_length_query = query_responses.shape[1]
        max_length_logits = max_length_query - context_length

        assert max_length_query > context_length
        assert query_responses.shape == (bs, max_length_query)
        assert logits.shape == (bs, max_length_logits, self.model.config.vocab_size)

    def test_single_batch_generation(self):
        batch = [
            self.tokenizer.apply_chat_template(example[:-1], add_generation_prompt=True, tokenize=False)
            for example in self.examples
        ]
        queries = self.tokenizer(batch, padding=True, return_tensors="pt")["input_ids"].to(self.device)
        bs, context_length = queries.shape

        query_responses, logits = batch_generation(
            self.model, queries, bs, self.tokenizer.pad_token_id, self.generation_config
        )

        max_length_query = query_responses.shape[1]
        max_length_logits = max_length_query - context_length

        assert max_length_query > context_length
        assert query_responses.shape == (bs, max_length_query)
        assert logits.shape == (bs, max_length_logits, self.model.config.vocab_size)


class BaseTester:
    class VHeadModelTester(TrlTestCase):
        all_model_names = None
        trl_model_class = None
        transformers_model_class = None

        def setup_method(self):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        def test_value_head(self):
            r"""
            Test if the v-head is added to the model successfully
            """
            for model_name in self.all_model_names:
                model = self.trl_model_class.from_pretrained(model_name)
                assert hasattr(model, "v_head")

        def test_value_head_shape(self):
            r"""
            Test if the v-head has the correct shape
            """
            for model_name in self.all_model_names:
                model = self.trl_model_class.from_pretrained(model_name)
                assert model.v_head.summary.weight.shape[0] == 1

        def test_value_head_init_random(self):
            r"""
            Test if the v-head has been randomly initialized. We can check that by making sure the bias is different
            than zeros by default.
            """
            for model_name in self.all_model_names:
                model = self.trl_model_class.from_pretrained(model_name)
                assert not torch.allclose(model.v_head.summary.bias, torch.zeros_like(model.v_head.summary.bias))

        def test_value_head_not_str(self):
            r"""
            Test if the v-head is added to the model successfully, by passing a non `PretrainedModel` as an argument to
            `from_pretrained`.
            """
            for model_name in self.all_model_names:
                pretrained_model = self.transformers_model_class.from_pretrained(model_name)
                model = self.trl_model_class.from_pretrained(pretrained_model)
                assert hasattr(model, "v_head")

        def test_from_save_trl(self):
            """
            Test if the model can be saved and loaded from a directory and get the same weights, including the
            additional modules (e.g. v_head)
            """
            for model_name in self.all_model_names:
                model = self.trl_model_class.from_pretrained(model_name)

                model.save_pretrained(self.tmp_dir)

                model_from_save = self.trl_model_class.from_pretrained(self.tmp_dir)

                # Check if the weights are the same
                for key in model_from_save.state_dict():
                    assert torch.allclose(model_from_save.state_dict()[key], model.state_dict()[key])

        def test_from_save_trl_sharded(self):
            """
            Test if the model can be saved and loaded from a directory and get the same weights - sharded case
            """
            for model_name in self.all_model_names:
                model = self.trl_model_class.from_pretrained(model_name)

                model.save_pretrained(self.tmp_dir)

                model_from_save = self.trl_model_class.from_pretrained(self.tmp_dir)

                # Check if the weights are the same
                for key in model_from_save.state_dict():
                    assert torch.allclose(model_from_save.state_dict()[key], model.state_dict()[key])

        def test_from_save_transformers_sharded(self):
            """
            Test if the model can be saved and loaded using transformers and get the same weights - sharded case
            """
            for model_name in self.all_model_names:
                transformers_model = self.trl_model_class.transformers_parent_class.from_pretrained(model_name)

                trl_model = self.trl_model_class.from_pretrained(model_name)

                trl_model.save_pretrained(self.tmp_dir, max_shard_size="1MB")
                transformers_model_from_save = self.trl_model_class.transformers_parent_class.from_pretrained(
                    self.tmp_dir
                )

                # Check if the weights are the same
                for key in transformers_model.state_dict():
                    assert torch.allclose(
                        transformers_model_from_save.state_dict()[key], transformers_model.state_dict()[key]
                    )

        def test_from_save_transformers(self):
            """
            Test if the model can be saved and loaded using transformers and get the same weights. We override the test
            of the super class to check if the weights are the same.
            """
            for model_name in self.all_model_names:
                transformers_model = self.trl_model_class.transformers_parent_class.from_pretrained(model_name)

                trl_model = self.trl_model_class.from_pretrained(model_name)

                trl_model.save_pretrained(self.tmp_dir)
                transformers_model_from_save = self.trl_model_class.transformers_parent_class.from_pretrained(
                    self.tmp_dir
                )

                # Check if the weights are the same
                for key in transformers_model.state_dict():
                    assert torch.allclose(
                        transformers_model_from_save.state_dict()[key], transformers_model.state_dict()[key]
                    )

                # Check if the trl model has the same keys as the transformers model
                # except the v_head
                for key in trl_model.state_dict():
                    if "v_head" not in key:
                        assert key in transformers_model.state_dict()
                        # check if the weights are the same
                        assert torch.allclose(trl_model.state_dict()[key], transformers_model.state_dict()[key])

                # check if they have the same modules
                assert set(transformers_model_from_save.state_dict().keys()) == set(
                    transformers_model.state_dict().keys()
                )


class TestCausalLMValueHeadModel(BaseTester.VHeadModelTester, TrlTestCase):
    """
    Testing suite for v-head models.
    """

    all_model_names = ALL_CAUSAL_LM_MODELS
    trl_model_class = AutoModelForCausalLMWithValueHead
    transformers_model_class = AutoModelForCausalLM

    def teardown_method(self):
        # free memory
        gc.collect()

    def test_inference(self):
        r"""
        Test if the model can be used for inference and outputs 3 values
        - logits, loss, and value states
        """
        EXPECTED_OUTPUT_SIZE = 3

        for model_name in self.all_model_names:
            model = self.trl_model_class.from_pretrained(model_name).to(self.device)
            input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], device=self.device)
            outputs = model(input_ids)

            # Check if the outputs are of the right size - here
            # we always output 3 values - logits, loss, and value states
            assert len(outputs) == EXPECTED_OUTPUT_SIZE

    def test_dropout_config(self):
        r"""
        Test if we instantiate a model by adding `summary_drop_prob` to the config it will be added to the v_head
        """
        for model_name in self.all_model_names:
            pretrained_model = self.transformers_model_class.from_pretrained(model_name)
            pretrained_model.config.summary_dropout_prob = 0.5
            model = self.trl_model_class.from_pretrained(pretrained_model)

            # Check if v head of the model has the same dropout as the config
            assert model.v_head.dropout.p == pretrained_model.config.summary_dropout_prob

    def test_dropout_kwargs(self):
        r"""
        Test if we instantiate a model by adding `summary_drop_prob` to the config it will be added to the v_head
        """
        for model_name in self.all_model_names:
            v_head_kwargs = {"summary_dropout_prob": 0.5}

            model = self.trl_model_class.from_pretrained(model_name, **v_head_kwargs)

            # Check if v head of the model has the same dropout as the config
            assert model.v_head.dropout.p == 0.5

            model = self.trl_model_class.from_pretrained(model_name, summary_dropout_prob=0.5)

            # Check if v head of the model has the same dropout as the config
            assert model.v_head.dropout.p == 0.5

    @pytest.mark.parametrize("model_name", ALL_CAUSAL_LM_MODELS)
    def test_generate(self, model_name):
        r"""
        Test if `generate` works for every model
        """
        generation_config = GenerationConfig(max_new_tokens=9)
        model = self.trl_model_class.from_pretrained(model_name).to(self.device)
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], device=self.device)

        # Just check if the generation works
        _ = model.generate(input_ids, generation_config=generation_config)

    def test_transformers_bf16_kwargs(self):
        r"""
        Test if the transformers kwargs are correctly passed. Here we check that loading a model in half precision
        works as expected, i.e. the weights of the `pretrained_model` attribute is loaded in half precision and you can
        run a dummy forward pass without any issue.
        """
        for model_name in self.all_model_names:
            trl_model = self.trl_model_class.from_pretrained(model_name, dtype=torch.bfloat16).to(self.device)

            lm_head_namings = ["lm_head", "embed_out", "output_layer"]

            assert any(hasattr(trl_model.pretrained_model, lm_head_naming) for lm_head_naming in lm_head_namings), (
                "Can't test the model because it doesn't have any of the expected lm_head namings"
            )

            for lm_head_naming in lm_head_namings:
                if hasattr(trl_model.pretrained_model, lm_head_naming):
                    assert getattr(trl_model.pretrained_model, lm_head_naming).weight.dtype == torch.bfloat16

            dummy_input = torch.LongTensor([[0, 1, 0, 1]]).to(self.device)

            # check dummy forward pass works in half precision
            _ = trl_model(dummy_input)

    @pytest.mark.skip(reason="This test needs to be run manually due to HF token issue.")
    def test_push_to_hub(self):
        for model_name in self.all_model_names:
            model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
            if "sharded" in model_name:
                model.push_to_hub(model_name + "-ppo", use_auth_token=True, max_shard_size="1MB")
            else:
                model.push_to_hub(model_name + "-ppo", use_auth_token=True)

            model_from_pretrained = AutoModelForCausalLMWithValueHead.from_pretrained(model_name + "-ppo")
            # check all keys
            assert model.state_dict().keys() == model_from_pretrained.state_dict().keys()

            for name, param in model.state_dict().items():
                assert torch.allclose(param, model_from_pretrained.state_dict()[name]), (
                    f"Parameter {name} is not the same after push_to_hub and from_pretrained"
                )


class TestSeq2SeqValueHeadModel(BaseTester.VHeadModelTester, TrlTestCase):
    """
    Testing suite for v-head models.
    """

    all_model_names = ALL_SEQ2SEQ_MODELS
    trl_model_class = AutoModelForSeq2SeqLMWithValueHead
    transformers_model_class = AutoModelForSeq2SeqLM

    def teardown_method(self):
        # free memory
        gc.collect()

    def test_inference(self):
        r"""
        Test if the model can be used for inference and outputs 3 values
        - logits, loss, and value states
        """
        EXPECTED_OUTPUT_SIZE = 3

        for model_name in self.all_model_names:
            model = self.trl_model_class.from_pretrained(model_name).to(self.device)
            input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], device=self.device)
            decoder_input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], device=self.device)
            outputs = model(input_ids, decoder_input_ids=decoder_input_ids)

            # Check if the outputs are of the right size - here
            # we always output 3 values - logits, loss, and value states
            assert len(outputs) == EXPECTED_OUTPUT_SIZE

    def test_dropout_config(self):
        r"""
        Test if we instantiate a model by adding `summary_drop_prob` to the config it will be added to the v_head
        """
        for model_name in self.all_model_names:
            pretrained_model = self.transformers_model_class.from_pretrained(model_name)
            pretrained_model.config.summary_dropout_prob = 0.5
            model = self.trl_model_class.from_pretrained(pretrained_model)

            # Check if v head of the model has the same dropout as the config
            assert model.v_head.dropout.p == pretrained_model.config.summary_dropout_prob

    def test_dropout_kwargs(self):
        r"""
        Test if we instantiate a model by adding `summary_drop_prob` to the config it will be added to the v_head
        """
        for model_name in self.all_model_names:
            v_head_kwargs = {"summary_dropout_prob": 0.5}

            model = self.trl_model_class.from_pretrained(model_name, **v_head_kwargs)

            # Check if v head of the model has the same dropout as the config
            assert model.v_head.dropout.p == 0.5

            model = self.trl_model_class.from_pretrained(model_name, summary_dropout_prob=0.5)

            # Check if v head of the model has the same dropout as the config
            assert model.v_head.dropout.p == 0.5

    @pytest.mark.parametrize("model_name", ALL_SEQ2SEQ_MODELS)
    def test_generate(self, model_name):
        r"""
        Test if `generate` works for every model
        """
        generation_config = GenerationConfig(max_new_tokens=9)
        model = self.trl_model_class.from_pretrained(model_name).to(self.device)
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], device=self.device)
        decoder_input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], device=self.device)

        # Just check if the generation works
        _ = model.generate(input_ids, decoder_input_ids=decoder_input_ids, generation_config=generation_config)

    @pytest.mark.skip(reason="This test needs to be run manually due to HF token issue.")
    def test_push_to_hub(self):
        for model_name in self.all_model_names:
            model = self.trl_model_class.from_pretrained(model_name)
            if "sharded" in model_name:
                model.push_to_hub(model_name + "-ppo", use_auth_token=True, max_shard_size="1MB")
            else:
                model.push_to_hub(model_name + "-ppo", use_auth_token=True)

            model_from_pretrained = self.trl_model_class.from_pretrained(model_name + "-ppo")
            # check all keys
            assert model.state_dict().keys() == model_from_pretrained.state_dict().keys()

            for name, param in model.state_dict().items():
                assert torch.allclose(param, model_from_pretrained.state_dict()[name]), (
                    f"Parameter {name} is not the same after push_to_hub and from_pretrained"
                )

    def test_transformers_bf16_kwargs(self):
        r"""
        Test if the transformers kwargs are correctly passed. Here we check that loading a model in half precision
        works as expected, i.e. the weights of the `pretrained_model` attribute is loaded in half precision and you can
        run a dummy forward pass without any issue.
        """
        for model_name in self.all_model_names:
            trl_model = self.trl_model_class.from_pretrained(model_name, dtype=torch.bfloat16).to(self.device)

            lm_head_namings = self.trl_model_class.lm_head_namings

            assert any(hasattr(trl_model.pretrained_model, lm_head_naming) for lm_head_naming in lm_head_namings)

            for lm_head_naming in lm_head_namings:
                if hasattr(trl_model.pretrained_model, lm_head_naming):
                    assert getattr(trl_model.pretrained_model, lm_head_naming).weight.dtype == torch.bfloat16

            dummy_input = torch.LongTensor([[0, 1, 0, 1]]).to(self.device)

            # check dummy forward pass works in half precision
            _ = trl_model(input_ids=dummy_input, decoder_input_ids=dummy_input)


@require_peft
class TestPeftModel(TrlTestCase):
    def setup_method(self):
        self.causal_lm_model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

    def test_create_peft_model(self):
        r"""
        Simply creates a peft model and checks that it can be loaded.
        """
        causal_lm_model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id)
        pretrained_model = get_peft_model(causal_lm_model, self.lora_config)

        _ = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model)

    def test_peft_requires_grad(self):
        r"""
        Check that the value head of the returned model has requires_grad=True.
        """
        causal_lm_model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id)
        pretrained_model = get_peft_model(causal_lm_model, self.lora_config)

        model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model)

        # Check that the value head has requires_grad=True
        assert model.v_head.summary.weight.requires_grad

    def test_check_peft_model_nb_trainable_params(self):
        r"""
        Check that the number of trainable parameters is correct.
        """
        causal_lm_model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id)
        pretrained_model = get_peft_model(causal_lm_model, self.lora_config)

        model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model)

        # Check that the number of trainable parameters is correct
        nb_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert nb_trainable_params == 905

        # Check that the number of trainable param for the non-peft model is correct
        non_peft_model = AutoModelForCausalLMWithValueHead.from_pretrained(self.causal_lm_model_id)
        nb_trainable_params = sum(p.numel() for p in non_peft_model.parameters() if p.requires_grad)
        assert nb_trainable_params == 2428641

    def test_create_peft_model_from_config(self):
        r"""
        Simply creates a peft model and checks that it can be loaded.
        """
        trl_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.causal_lm_model_id, peft_config=self.lora_config
        )
        # Check that the number of trainable parameters is correct
        nb_trainable_params = sum(p.numel() for p in trl_model.parameters() if p.requires_grad)
        assert nb_trainable_params == 905

        causal_lm_model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id)
        trl_model = AutoModelForCausalLMWithValueHead.from_pretrained(causal_lm_model, peft_config=self.lora_config)
        # Check that the number of trainable parameters is correct
        nb_trainable_params = sum(p.numel() for p in trl_model.parameters() if p.requires_grad)
        assert nb_trainable_params == 905

    @require_bitsandbytes
    @require_torch_gpu_if_bnb_not_multi_backend_enabled
    def test_create_bnb_peft_model_from_config(self):
        r"""
        Simply creates a peft model and checks that it can be loaded.
        """
        from bitsandbytes.nn import Linear8bitLt
        from transformers import BitsAndBytesConfig

        trl_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.causal_lm_model_id,
            peft_config=self.lora_config,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        )
        # Check that the number of trainable parameters is correct
        nb_trainable_params = sum(p.numel() for p in trl_model.parameters() if p.requires_grad)
        assert nb_trainable_params == 905
        assert isinstance(trl_model.pretrained_model.model.model.layers[0].mlp.gate_proj, Linear8bitLt)

        causal_lm_model = AutoModelForCausalLM.from_pretrained(
            self.causal_lm_model_id, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto"
        )
        trl_model = AutoModelForCausalLMWithValueHead.from_pretrained(causal_lm_model, peft_config=self.lora_config)
        # Check that the number of trainable parameters is correct
        nb_trainable_params = sum(p.numel() for p in trl_model.parameters() if p.requires_grad)
        assert nb_trainable_params == 905
        assert isinstance(trl_model.pretrained_model.model.model.layers[0].mlp.gate_proj, Linear8bitLt)

    def test_save_pretrained_peft(self):
        r"""
        Check that the model can be saved and loaded properly.
        """
        causal_lm_model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id)
        pretrained_model = get_peft_model(causal_lm_model, self.lora_config)

        model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model)

        model.save_pretrained(self.tmp_dir)

        # check that the files `adapter_model.safetensors` and `adapter_config.json` are in the directory
        assert os.path.isfile(f"{self.tmp_dir}/adapter_model.safetensors"), (
            f"{self.tmp_dir}/adapter_model.safetensors does not exist"
        )
        assert os.path.exists(f"{self.tmp_dir}/adapter_config.json"), (
            f"{self.tmp_dir}/adapter_config.json does not exist"
        )

        # check also for `pytorch_model.bin` and make sure it only contains `v_head` weights
        assert os.path.exists(f"{self.tmp_dir}/pytorch_model.bin"), f"{self.tmp_dir}/pytorch_model.bin does not exist"

        # check that only keys that starts with `v_head` are in the dict
        maybe_v_head = torch.load(f"{self.tmp_dir}/pytorch_model.bin", weights_only=True)
        assert all(k.startswith("v_head") for k in maybe_v_head.keys()), (
            f"keys in {self.tmp_dir}/pytorch_model.bin do not start with `v_head`"
        )

        model_from_pretrained = AutoModelForCausalLMWithValueHead.from_pretrained(self.tmp_dir)

        # check all the weights are the same
        for p1, p2 in zip(model.named_parameters(), model_from_pretrained.named_parameters(), strict=True):
            assert torch.allclose(p1[1], p2[1]), f"{p1[0]} != {p2[0]}"

    def test_load_pretrained_peft(self):
        r"""
        Check that the model saved with peft class interface can be loaded properly.
        """
        causal_lm_model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id)
        pretrained_model = get_peft_model(causal_lm_model, self.lora_config)

        model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model)

        pretrained_model.save_pretrained(self.tmp_dir)
        model_from_pretrained = AutoModelForCausalLMWithValueHead.from_pretrained(self.tmp_dir)

        # check that the files `adapter_model.safetensors` and `adapter_config.json` are in the directory
        assert os.path.isfile(f"{self.tmp_dir}/adapter_model.safetensors"), (
            f"{self.tmp_dir}/adapter_model.safetensors does not exist"
        )
        assert os.path.exists(f"{self.tmp_dir}/adapter_config.json"), (
            f"{self.tmp_dir}/adapter_config.json does not exist"
        )

        # check all the weights are the same
        for p1, p2 in zip(model.named_parameters(), model_from_pretrained.named_parameters(), strict=True):
            if p1[0] not in ["v_head.summary.weight", "v_head.summary.bias"]:
                assert torch.allclose(p1[1], p2[1]), f"{p1[0]} != {p2[0]}"

    def test_continue_training_peft_model(self):
        r"""
        Load peft and checks that it can continue training.
        """
        causal_lm_model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id)
        pretrained_model = get_peft_model(causal_lm_model, self.lora_config)

        pretrained_model.save_pretrained(self.tmp_dir)
        # set is_trainable to True
        model = AutoModelForCausalLMWithValueHead.from_pretrained(self.tmp_dir, is_trainable=True)
        # Check that the number of trainable parameters is correct
        nb_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert nb_trainable_params == 905


class TestCore(TrlTestCase):
    """
    A wrapper class for testing core utils functions
    """

    def setup_method(self):
        self.test_input = torch.Tensor([1, 2, 3, 4])
        self.test_mask = torch.Tensor([0, 1, 1, 0])
        self.test_input_unmasked = self.test_input[1:3]

    def test_masked_mean(self):
        assert torch.mean(self.test_input_unmasked) == masked_mean(self.test_input, self.test_mask)

    def test_masked_var(self):
        assert torch.var(self.test_input_unmasked) == masked_var(self.test_input, self.test_mask)

    def test_masked_whiten(self):
        def whiten(values: torch.Tensor) -> torch.Tensor:
            mean, var = torch.mean(values), torch.var(values)
            return (values - mean) * torch.rsqrt(var + 1e-8)

        whiten_unmasked = whiten(self.test_input_unmasked)
        whiten_masked = masked_whiten(self.test_input, self.test_mask)[1:3]
        diffs = (whiten_unmasked - whiten_masked).sum()
        assert abs(diffs.item()) < 0.00001


class TestPPOTrainer(TrlTestCase):
    def setup_method(self):
        # Set up the models and tokenizer using the test model
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.ref_model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side="left")
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # Add reward and value models as in ppo.py
        reward_model_id = "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5"
        self.value_model = AutoModelForSequenceClassification.from_pretrained(reward_model_id, num_labels=1)
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_id, num_labels=1)

        # Load dataset
        raw_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

        def tokenize(example, tokenizer):
            tokenized = tokenizer(text=example["prompt"])
            if tokenizer.eos_token_id is not None and tokenized["input_ids"][-1] != tokenizer.eos_token_id:
                tokenized["input_ids"] = tokenized["input_ids"] + [tokenizer.eos_token_id]
                tokenized["attention_mask"] = tokenized["attention_mask"] + [1]
            return tokenized

        self.raw_dataset = raw_dataset.map(tokenize, fn_kwargs={"tokenizer": self.tokenizer}, remove_columns="prompt")

    def test_basic_training(self):
        """Test basic PPO training configuration and verify model updates."""
        # Capture initial weights
        initial_critic_weights = {}
        initial_policy_weights = {}
        for name, param in self.value_model.named_parameters():
            initial_critic_weights[name] = param.clone().detach()
        for name, param in self.model.named_parameters():
            initial_policy_weights[name] = param.clone().detach()

        # Configure training args similar to example script
        training_args = PPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=2,
            num_ppo_epochs=2,  # Decrease number of PPO epochs to speed up test
            report_to="none",
        )

        # Create trainer
        trainer = PPOTrainer(
            args=training_args,
            processing_class=self.tokenizer,
            model=self.model,
            ref_model=self.ref_model,
            reward_model=self.reward_model,
            value_model=self.value_model,
            train_dataset=self.raw_dataset["train"],
            eval_dataset=self.raw_dataset["test"],
        )

        # Train
        trainer.train()

        # Check if critic weights have been updated
        critic_weights_updated = False
        for name, param in trainer.model.value_model.named_parameters():
            if not torch.allclose(initial_critic_weights[name], param.to("cpu")):
                critic_weights_updated = True
                break

        # Check if policy weights have been updated
        policy_weights_updated = False
        for name, param in trainer.model.policy.named_parameters():
            if not torch.allclose(initial_policy_weights[name], param.to("cpu")):
                policy_weights_updated = True
                break

        assert critic_weights_updated, "Critic weights were not updated during training"
        assert policy_weights_updated, "Policy weights were not updated during training"

    @require_peft
    def test_peft_training(self):
        """Test PPO training with PEFT configuration and verify model updates."""
        # Capture initial weights
        initial_critic_weights = {}
        initial_policy_weights = {}
        for name, param in self.value_model.named_parameters():
            initial_critic_weights[name] = param.clone().detach()
        for name, param in self.model.named_parameters():
            initial_policy_weights[name] = param.clone().detach()

        # Configure training args
        training_args = PPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=2,
            num_ppo_epochs=2,  # Decrease number of PPO epochs to speed up test
            report_to="none",
        )

        # Configure PEFT
        peft_config = LoraConfig(
            r=32,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Create trainer with PEFT
        trainer = PPOTrainer(
            args=training_args,
            processing_class=self.tokenizer,
            model=self.model,
            ref_model=None,
            reward_model=self.reward_model,
            value_model=self.value_model,
            train_dataset=self.raw_dataset["train"],
            eval_dataset=self.raw_dataset["test"],
            peft_config=peft_config,
        )

        # Train
        trainer.train()

        # Check if critic weights have been updated
        critic_weights_updated = False
        for name, param in trainer.model.value_model.named_parameters():
            if name in initial_critic_weights and not torch.allclose(initial_critic_weights[name], param.to("cpu")):
                critic_weights_updated = True
                break

        # Check if policy weights have been updated - for PEFT we check the LoRA weights
        policy_weights_updated = False
        for name, param in trainer.model.policy.named_parameters():
            if "lora" in name.lower() and param.requires_grad:  # Only check LoRA weights
                # New weights should be non-zero if they've been updated
                if not torch.allclose(param, torch.zeros_like(param)):
                    policy_weights_updated = True
                    break

        assert critic_weights_updated, "Critic weights were not updated during training"
        assert policy_weights_updated, "Policy LoRA weights were not updated during training"
