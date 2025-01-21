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

# This script generates tiny models used in the TRL library for unit tests. It pushes them to the Hub under the
# `trl-internal-testing` organization.
# This script is meant to be run when adding new tiny model to the TRL library.

from huggingface_hub import HfApi, ModelCard
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    BartConfig,
    BartModel,
    BloomConfig,
    BloomForCausalLM,
    CLIPVisionConfig,
    CohereConfig,
    CohereForCausalLM,
    DbrxConfig,
    DbrxForCausalLM,
    FalconMambaConfig,
    FalconMambaForCausalLM,
    Gemma2Config,
    Gemma2ForCausalLM,
    GemmaConfig,
    GemmaForCausalLM,
    GPT2Config,
    GPT2LMHeadModel,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    Idefics2Config,
    Idefics2ForConditionalGeneration,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlavaConfig,
    LlavaForConditionalGeneration,
    LlavaNextConfig,
    LlavaNextForConditionalGeneration,
    MistralConfig,
    MistralForCausalLM,
    OPTConfig,
    OPTForCausalLM,
    PaliGemmaConfig,
    PaliGemmaForConditionalGeneration,
    Phi3Config,
    Phi3ForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2ForSequenceClassification,
    SiglipVisionConfig,
    T5Config,
    T5ForConditionalGeneration,
)
from transformers.models.idefics2.configuration_idefics2 import Idefics2VisionConfig


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


def push_to_hub(model, tokenizer, prefix=None, suffix=None):
    model_class_name = model.__class__.__name__
    content = MODEL_CARD.format(model_class_name=model_class_name)
    model_card = ModelCard(content)
    if prefix is not None:
        model_class_name = f"{prefix}-{model_class_name}"
    repo_id = f"{ORGANIZATION}/{model_class_name}"
    if suffix is not None:
        repo_id += f"-{suffix}"

    if api.repo_exists(repo_id):
        print(f"Model {repo_id} already exists, skipping")
    else:
        model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)
        model_card.push_to_hub(repo_id)


# Decoder models
for model_id, config_class, model_class, suffix in [
    ("bigscience/bloomz-560m", BloomConfig, BloomForCausalLM, None),
    ("CohereForAI/aya-expanse-8b", CohereConfig, CohereForCausalLM, None),
    ("databricks/dbrx-instruct", DbrxConfig, DbrxForCausalLM, None),
    ("tiiuae/falcon-7b-instruct", FalconMambaConfig, FalconMambaForCausalLM, None),
    ("google/gemma-2-2b-it", Gemma2Config, Gemma2ForCausalLM, None),
    ("google/gemma-7b-it", GemmaConfig, GemmaForCausalLM, None),
    ("openai-community/gpt2", GPT2Config, GPT2LMHeadModel, None),
    ("EleutherAI/pythia-14m", GPTNeoXConfig, GPTNeoXForCausalLM, None),
    ("meta-llama/Meta-Llama-3-8B-Instruct", LlamaConfig, LlamaForCausalLM, "3"),
    ("meta-llama/Llama-3.1-8B-Instruct", LlamaConfig, LlamaForCausalLM, "3.1"),
    ("meta-llama/Llama-3.2-1B-Instruct", LlamaConfig, LlamaForCausalLM, "3.2"),
    ("mistralai/Mistral-7B-Instruct-v0.1", MistralConfig, MistralForCausalLM, "0.1"),
    ("mistralai/Mistral-7B-Instruct-v0.2", MistralConfig, MistralForCausalLM, "0.2"),
    ("facebook/opt-1.3b", OPTConfig, OPTForCausalLM, None),
    ("microsoft/Phi-3.5-mini-instruct", Phi3Config, Phi3ForCausalLM, None),
    ("Qwen/Qwen2.5-32B-Instruct", Qwen2Config, Qwen2ForCausalLM, "2.5"),
    ("Qwen/Qwen2.5-Coder-0.5B", Qwen2Config, Qwen2ForCausalLM, "2.5-Coder"),
]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = config_class(
        vocab_size=tokenizer.vocab_size + len(tokenizer.added_tokens_encoder.keys()),
        hidden_size=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        intermediate_size=32,
    )
    model = model_class(config)
    push_to_hub(model, tokenizer, "tiny", suffix)


# A slightly bigger model, required for vLLM testing
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")
config = Qwen2Config(
    vocab_size=tokenizer.vocab_size + len(tokenizer.added_tokens_encoder.keys()),
    hidden_size=128,  # increase hidden size so that hidden_size // num_attention_heads = 32, required for vLLM
    num_attention_heads=4,
    num_key_value_heads=2,
    num_hidden_layers=2,
    intermediate_size=32,
)
model = Qwen2ForCausalLM(config)
push_to_hub(model, tokenizer, "small", "2.5")


# Reward models
for model_id, config_class, model_class, suffix in [
    ("meta-llama/Llama-3.2-1B-Instruct", LlamaConfig, LlamaForSequenceClassification, "3.2"),
    ("Qwen/Qwen2.5-32B-Instruct", Qwen2Config, Qwen2ForSequenceClassification, "2.5"),
]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = config_class(
        vocab_size=tokenizer.vocab_size + len(tokenizer.added_tokens_encoder.keys()),
        hidden_size=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        intermediate_size=32,
        num_labels=1,
    )
    model = model_class(config)
    push_to_hub(model, tokenizer, "tiny", suffix)


# Encoder-decoder models
for model_id, config_class, model_class, suffix in [
    ("google/flan-t5-small", T5Config, T5ForConditionalGeneration, None),
    ("facebook/bart-base", BartConfig, BartModel, None),
]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = config_class(
        vocab_size=tokenizer.vocab_size + len(tokenizer.added_tokens_encoder.keys()),
        d_model=16,
        encoder_layers=2,
        decoder_layers=2,
        d_kv=2,
        d_ff=64,
        num_layers=6,
        num_heads=8,
        decoder_start_token_id=0,
        is_encoder_decoder=True,
    )
    model = model_class(config)
    push_to_hub(model, tokenizer, "tiny", suffix)


# Vision Language Models
# fmt: off
for model_id, config_class, text_config_class, vision_config_class, model_class in [
    ("HuggingFaceM4/idefics2-8b", Idefics2Config, MistralConfig, Idefics2VisionConfig, Idefics2ForConditionalGeneration),
    ("llava-hf/llava-1.5-7b-hf", LlavaConfig, LlamaConfig, CLIPVisionConfig, LlavaForConditionalGeneration),
    ("llava-hf/llava-v1.6-mistral-7b-hf", LlavaNextConfig, MistralConfig, CLIPVisionConfig, LlavaNextForConditionalGeneration),
    ("google/paligemma-3b-pt-224", PaliGemmaConfig, GemmaConfig, SiglipVisionConfig, PaliGemmaForConditionalGeneration),
]:
# fmt: on
    processor = AutoProcessor.from_pretrained(model_id)
    kwargs = {}
    if config_class == PaliGemmaConfig:
        kwargs["projection_dim"] = 8
    vision_kwargs = {}
    if vision_config_class in [CLIPVisionConfig, SiglipVisionConfig]:
        vision_kwargs["projection_dim"] = 8
    if vision_config_class == CLIPVisionConfig:
        vision_kwargs["image_size"] = 336
        vision_kwargs["patch_size"] = 14
    config = config_class(
        text_config=text_config_class(
            vocab_size=processor.tokenizer.vocab_size + len(processor.tokenizer.added_tokens_encoder),
            hidden_size=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
            intermediate_size=32,
        ),
        vision_config=vision_config_class(
            hidden_size=8,
            num_attention_heads=4,
            num_hidden_layers=2,
            intermediate_size=32,
            **vision_kwargs,
        ),
        **kwargs,
    )
    model = model_class(config)
    push_to_hub(model, processor, "tiny")
