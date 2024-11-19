# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from huggingface_hub import ModelCard
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    BartConfig,
    BartModel,
    BloomConfig,
    BloomForCausalLM,
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
    SiglipVisionConfig,
    T5Config,
    T5ForConditionalGeneration,
)
from transformers.models.idefics2.configuration_idefics2 import Idefics2VisionConfig


ORGANIZATION = "qgallouedec"

MODEL_CARD = """
---
library_name: transformers
tags: [trl]
---

# Tiny {model_class_name}

This is a minimal model built for unit tests in the [TRL](https://github.com/huggingface/trl) library.
"""


def push_to_hub(model, tokenizer, suffix=None):
    model_class_name = model.__class__.__name__
    content = MODEL_CARD.format(model_class_name=model_class_name)
    model_card = ModelCard(content)
    repo_id = f"{ORGANIZATION}/tiny-{model_class_name}"
    if suffix is not None:
        repo_id += f"-{suffix}"
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
    ("mistralai/Mistral-7B-Instruct-v0.3", MistralConfig, MistralForCausalLM, "0.3"),
    ("facebook/opt-1.3b", OPTConfig, OPTForCausalLM, None),
    ("microsoft/Phi-3.5-mini-instruct", Phi3Config, Phi3ForCausalLM, None),
    ("Qwen/Qwen2.5-32B-Instruct", Qwen2Config, Qwen2ForCausalLM, "2.5"),
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
    push_to_hub(model, tokenizer, suffix)


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
    push_to_hub(model, tokenizer, suffix)


# Vision Language Models

# Idefics2
processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")
config = Idefics2Config(
    text_config=MistralConfig(
        vocab_size=processor.tokenizer.vocab_size + len(processor.tokenizer.added_tokens_encoder),
        hidden_size=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        intermediate_size=32,
    ),
    vision_config=Idefics2VisionConfig(
        hidden_size=8,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=32,
    ),
)
model = Idefics2ForConditionalGeneration(config)
push_to_hub(model, processor)


# PaliGemma
processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")
# PaliGemma is not meant for chat, but we add a chat template for testing purposes
processor.chat_template = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' }}{% for content in message['content'] %}{% if content['type'] == 'text' %}{{ content['text'] | trim }}{% endif %}{% endfor %}{{ '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model'}}{% endif %}"
config = PaliGemmaConfig(
    projection_dim=8,
    text_config=GemmaConfig(
        vocab_size=processor.tokenizer.vocab_size + len(processor.tokenizer.added_tokens_encoder),
        hidden_size=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        intermediate_size=32,
    ),
    vision_config=SiglipVisionConfig(
        hidden_size=8,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=32,
        projection_dim=8,
    ),
)
model = PaliGemmaForConditionalGeneration(config)
push_to_hub(model, processor)