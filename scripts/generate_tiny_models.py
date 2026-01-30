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

# This script generates tiny models used in the TRL library for unit tests. It pushes them to the Hub under the
# `trl-internal-testing` organization.
# This script is meant to be run when adding new tiny model to the TRL library.

import torch
from huggingface_hub import HfApi, ModelCard
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    BartModel,
    CohereConfig,
    CohereForCausalLM,
    DeepseekV3Config,
    DeepseekV3ForCausalLM,
    FalconMambaConfig,
    FalconMambaForCausalLM,
    Gemma2Config,
    Gemma2ForCausalLM,
    Gemma3ForConditionalGeneration,
    GemmaConfig,
    GemmaForCausalLM,
    GenerationConfig,
    GPT2Config,
    GPT2LMHeadModel,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    GPTNeoXForSequenceClassification,
    GptOssConfig,
    GptOssForCausalLM,
    Idefics2Config,
    Idefics2ForConditionalGeneration,
    Idefics3ForConditionalGeneration,
    InternVLForConditionalGeneration,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    MistralConfig,
    MistralForCausalLM,
    OPTConfig,
    OPTForCausalLM,
    PaliGemmaForConditionalGeneration,
    Phi3Config,
    Phi3ForCausalLM,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2ForSequenceClassification,
    Qwen2VLConfig,
    Qwen2VLForConditionalGeneration,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3ForSequenceClassification,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
    Qwen3MoeForSequenceClassification,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
    SmolVLMForConditionalGeneration,
    T5ForConditionalGeneration,
)


ORGANIZATION = "trl-internal-testing"

MODEL_CARD = """
---
library_name: transformers
tags: [trl]
---

# Tiny {model_class_name}

This is a minimal model built for unit tests in the [TRL](https://github.com/huggingface/trl) library.
"""


api = HfApi()


def push_to_hub(model, tokenizer, generation_config, prefix=None, suffix=None, force=False):
    model_class_name = model.__class__.__name__
    content = MODEL_CARD.format(model_class_name=model_class_name)
    model_card = ModelCard(content)
    if prefix is not None:
        model_class_name = f"{prefix}-{model_class_name}"
    repo_id = f"{ORGANIZATION}/{model_class_name}"
    if suffix is not None:
        repo_id += f"-{suffix}"

    if api.repo_exists(repo_id) and not force:
        print(f"Model {repo_id} already exists, skipping")
    else:
        model.push_to_hub(repo_id)
        model_card.push_to_hub(repo_id)
        if tokenizer is not None:
            tokenizer.push_to_hub(repo_id)
        if generation_config is not None:
            generation_config.push_to_hub(repo_id)


def init_weights_tiny_model(model):
    """
    Initialize tiny test models to avoid NaNs from uninitialized weights.

    Uses safe defaults:
      - Linear/Conv1d: Xavier uniform (weights), zero (biases)
      - Embedding: Normal(0, 0.02)
      - LayerNorm: Ones (weights), zero (biases)

    Args:
        model: PyTorch model (modified in-place)
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # Attention/MLP projections → Xavier or Normal
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            nn.init.xavier_uniform_(module.weight)

        elif isinstance(module, nn.Embedding):
            # Token embeddings → GPT-style Normal
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif isinstance(module, nn.LayerNorm):
            # LayerNorm weights always 1, bias 0
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Conv1d):
            # Convolutional layers → Xavier or Normal
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            nn.init.xavier_uniform_(module.weight)


# Decoder models
for model_id, config_class, model_class, dtype, suffix in [
    # ("bigscience/bloomz-560m", BloomConfig, BloomForCausalLM, None),  # loading fails with this model, see https://huggingface.co/bigscience/bloomz-560m/discussions/14
    ("CohereLabs/aya-expanse-8b", CohereConfig, CohereForCausalLM, torch.float16, None),
    ("deepseek-ai/DeepSeek-R1", DeepseekV3Config, DeepseekV3ForCausalLM, torch.bfloat16, None),
    # It's important to have R1-0528 as it doesn't have the same chat template
    ("deepseek-ai/DeepSeek-R1-0528", DeepseekV3Config, DeepseekV3ForCausalLM, torch.bfloat16, "0528"),
    ("tiiuae/falcon-7b-instruct", FalconMambaConfig, FalconMambaForCausalLM, torch.bfloat16, None),
    ("google/gemma-2-2b-it", Gemma2Config, Gemma2ForCausalLM, torch.bfloat16, None),
    ("google/gemma-7b-it", GemmaConfig, GemmaForCausalLM, torch.bfloat16, None),
    ("openai-community/gpt2", GPT2Config, GPT2LMHeadModel, torch.float32, None),
    ("EleutherAI/pythia-14m", GPTNeoXConfig, GPTNeoXForCausalLM, torch.float16, None),
    ("meta-llama/Meta-Llama-3-8B-Instruct", LlamaConfig, LlamaForCausalLM, torch.bfloat16, "3"),
    ("meta-llama/Llama-3.1-8B-Instruct", LlamaConfig, LlamaForCausalLM, torch.bfloat16, "3.1"),
    ("meta-llama/Llama-3.2-1B-Instruct", LlamaConfig, LlamaForCausalLM, torch.bfloat16, "3.2"),
    ("mistralai/Mistral-7B-Instruct-v0.1", MistralConfig, MistralForCausalLM, torch.bfloat16, "0.1"),
    ("mistralai/Mistral-7B-Instruct-v0.2", MistralConfig, MistralForCausalLM, torch.bfloat16, "0.2"),
    ("facebook/opt-1.3b", OPTConfig, OPTForCausalLM, torch.float16, None),
    ("microsoft/Phi-3.5-mini-instruct", Phi3Config, Phi3ForCausalLM, torch.bfloat16, None),
    ("Qwen/Qwen2.5-32B-Instruct", Qwen2Config, Qwen2ForCausalLM, torch.bfloat16, "2.5"),
    ("Qwen/Qwen2.5-Coder-0.5B", Qwen2Config, Qwen2ForCausalLM, torch.bfloat16, "2.5-Coder"),
    ("Qwen/Qwen3-8B", Qwen3Config, Qwen3ForCausalLM, torch.bfloat16, None),
]:
    revision = "refs/pr/14" if model_id == "Qwen/Qwen3-8B" else "main"  # chat template with {% generation %}
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    generation_config = GenerationConfig.from_pretrained(model_id, revision=revision)
    config = config_class(
        vocab_size=len(tokenizer.vocab),
        hidden_size=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        intermediate_size=32,
    )
    model = model_class(config).to(dtype=dtype)
    init_weights_tiny_model(model)
    push_to_hub(model, tokenizer, generation_config, "tiny", suffix)

# MoE models
for model_id, config_class, model_class, dtype, suffix in [
    ("Qwen/Qwen3-30B-A3B", Qwen3MoeConfig, Qwen3MoeForCausalLM, torch.bfloat16, None),
    ("openai/gpt-oss-20b", GptOssConfig, GptOssForCausalLM, torch.bfloat16, None),
]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    generation_config = GenerationConfig.from_pretrained(model_id)
    config = config_class(
        vocab_size=len(tokenizer.vocab),
        hidden_size=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
    )
    model = model_class(config).to(dtype=dtype)
    init_weights_tiny_model(model)
    push_to_hub(model, tokenizer, generation_config, "tiny", suffix)

# Two slightly bigger models, required for vLLM testing
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")
generation_config = GenerationConfig.from_pretrained("Qwen/Qwen2.5-32B-Instruct")
config = Qwen2Config(
    vocab_size=len(tokenizer.vocab),
    hidden_size=128,  # increase hidden size so that hidden_size // num_attention_heads = 32, required for vLLM
    num_attention_heads=4,
    num_key_value_heads=2,
    num_hidden_layers=2,
    intermediate_size=32,
)
model = Qwen2ForCausalLM(config).to(dtype=torch.bfloat16)
push_to_hub(model, tokenizer, generation_config, "small", "2.5")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
generation_config = GenerationConfig.from_pretrained("Qwen/Qwen3-4B")
config = Qwen3Config(
    vocab_size=len(tokenizer.vocab),
    hidden_size=128,  # increase hidden size so that hidden_size // num_attention_heads = 32, required for vLLM
    num_attention_heads=4,
    num_key_value_heads=2,
    num_hidden_layers=2,
    intermediate_size=32,
)
model = Qwen3ForCausalLM(config).to(dtype=torch.bfloat16)
push_to_hub(model, tokenizer, generation_config, "small")

# Reward models
for model_id, model_class, dtype, suffix in [
    ("EleutherAI/pythia-14m", GPTNeoXForSequenceClassification, torch.bfloat16, None),
    ("meta-llama/Llama-3.2-1B-Instruct", LlamaForSequenceClassification, torch.bfloat16, "3.2"),
    ("Qwen/Qwen2.5-32B-Instruct", Qwen2ForSequenceClassification, torch.bfloat16, "2.5"),
    ("Qwen/Qwen3-4B", Qwen3ForSequenceClassification, torch.bfloat16, None),
]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    generation_config = GenerationConfig.from_pretrained(model_id)
    kwargs = {
        "num_labels": 1,
        "hidden_size": 16,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 2,
        "intermediate_size": 32,
    }
    config = AutoConfig.from_pretrained(model_id, **kwargs)
    # Bug in transformers: it ignores num_hidden_layers to build layer_types
    if model_id in ("Qwen/Qwen2.5-32B-Instruct", "Qwen/Qwen3-4B"):
        config.layer_types = config.layer_types[:2]
    model = model_class(config).to(dtype=dtype)
    init_weights_tiny_model(model)
    push_to_hub(model, tokenizer, generation_config, "tiny", suffix)

# MoE Reward models
for model_id, model_class, dtype, suffix in [
    ("Qwen/Qwen3-30B-A3B", Qwen3MoeForSequenceClassification, torch.bfloat16, None),
]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    generation_config = GenerationConfig.from_pretrained(model_id)
    kwargs = {
        "num_labels": 1,
        "hidden_size": 16,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 2,
        "intermediate_size": 32,
        "num_experts": 4,
        "num_experts_per_tok": 2,
    }
    config = AutoConfig.from_pretrained(model_id, **kwargs)
    model = model_class(config).to(dtype=dtype)
    push_to_hub(model, tokenizer, generation_config, "tiny", suffix)


# Encoder-decoder models
for model_id, model_class, dtype, suffix in [
    ("facebook/bart-base", BartModel, torch.float32, None),
    ("google/flan-t5-small", T5ForConditionalGeneration, torch.float32, None),
]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    generation_config = GenerationConfig.from_pretrained(model_id) if model_id != "facebook/bart-base" else None
    config = AutoConfig.from_pretrained(model_id)
    config.d_model = 24
    model = model_class(config).to(dtype=dtype)
    push_to_hub(model, tokenizer, generation_config, "tiny", suffix)


# Vision Language Models
for model_id, model_class, dtype in [
    ("google/gemma-3-4b-it", Gemma3ForConditionalGeneration, torch.bfloat16),
    ("google/paligemma-3b-pt-224", PaliGemmaForConditionalGeneration, torch.float32),
    ("HuggingFaceM4/idefics2-8b", Idefics2ForConditionalGeneration, torch.float32),
    ("HuggingFaceM4/Idefics3-8B-Llama3", Idefics3ForConditionalGeneration, torch.bfloat16),
    ("HuggingFaceTB/SmolVLM2-2.2B-Instruct", SmolVLMForConditionalGeneration, torch.float32),
    ("llava-hf/llava-1.5-7b-hf", LlavaForConditionalGeneration, torch.float16),
    # Original model dtype is float16, but it triggers CUDA device side assert error (see GH-4741):
    ("llava-hf/llava-v1.6-mistral-7b-hf", LlavaNextForConditionalGeneration, torch.bfloat16),
    ("OpenGVLab/InternVL3-8B-hf", InternVLForConditionalGeneration, torch.bfloat16),
    ("Qwen/Qwen2-VL-2B-Instruct", Qwen2VLForConditionalGeneration, torch.bfloat16),
    ("Qwen/Qwen2.5-VL-3B-Instruct", Qwen2_5_VLForConditionalGeneration, torch.bfloat16),
    ("Qwen/Qwen3-VL-2B-Instruct", Qwen3VLForConditionalGeneration, torch.bfloat16),
]:
    processor = AutoProcessor.from_pretrained(model_id)
    generation_config = GenerationConfig.from_pretrained(model_id)

    text_config = {
        "num_hidden_layers": 2,
        "hidden_size": 16,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "layer_types": None,  # Set it automatically from num_hidden_layers
    }
    vision_config = {
        "num_hidden_layers": 2,
        "hidden_size": 16,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "embed_dim": 64,
    }
    kwargs = {}

    if issubclass(model_class.config_class, (Qwen2VLConfig, Qwen2_5_VLConfig)):
        text_config["rope_scaling"] = {"type": "default", "mrope_section": [1, 1], "rope_type": "default"}
        vision_config["depth"] = 2
        # Different dict object from text_config; see GH-4101 and transformers#41020
        kwargs["rope_scaling"] = {"type": "default", "mrope_section": [1, 1], "rope_type": "default"}

    if issubclass(model_class.config_class, Qwen2_5_VLConfig):
        vision_config["out_hidden_size"] = 16
        # Different dict object at the config root; see GH-4101 and transformers#41020
        kwargs["num_hidden_layers"] = 2
        kwargs["hidden_size"] = 16
        kwargs["num_attention_heads"] = 4

    if issubclass(model_class.config_class, Idefics2Config):
        kwargs["perceiver_config"] = {"hidden_size": 16}

    if issubclass(model_class.config_class, Qwen3VLConfig):
        # So hasattr(config, "layer_types") is False
        # See: https://github.com/huggingface/transformers/blob/fe5ca9ddaa07fac2872407e75c7a7661216ac956/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L420
        del text_config["layer_types"]
        # "mrope_section" needs 3 elements: for dim, offset in enumerate((1, 2), start=1): mrope_section[dim]
        # See: https://github.com/huggingface/transformers/blob/fe5ca9ddaa07fac2872407e75c7a7661216ac956/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L361
        text_config["rope_scaling"] = {"mrope_interleaved": True, "mrope_section": [2, 2, 2], "rope_type": "default"}
        vision_config["depth"] = 2
        vision_config["out_hidden_size"] = 16

    if model_id == "llava-hf/llava-v1.6-mistral-7b-hf":
        # Hotfix: llava-hf/llava-v1.6-mistral-7b-hf mistakesly sets text_config.dtype to "bfloat16".
        # See https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf/discussions/46
        text_config["dtype"] = None

    config = AutoConfig.from_pretrained(model_id, text_config=text_config, vision_config=vision_config, **kwargs)
    model = model_class(config).to(dtype=dtype)
    push_to_hub(model, processor, generation_config, "tiny")

# PEFT models
model = Qwen3ForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen3ForCausalLM", dtype="auto")
model = get_peft_model(model, LoraConfig())
generation_config = GenerationConfig.from_pretrained("trl-internal-testing/tiny-Qwen3ForCausalLM")
push_to_hub(model, None, None, "tiny")

# Same model, but different weights
model = Qwen3ForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen3ForCausalLM", dtype="auto")
model = get_peft_model(model, LoraConfig())
generation_config = GenerationConfig.from_pretrained("trl-internal-testing/tiny-Qwen3ForCausalLM")
push_to_hub(model, None, None, "tiny", "2")
