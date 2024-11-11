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
    BartForCausalLM,
    BartTokenizerFast,
    CohereConfig,
    CohereForCausalLM,
    CohereTokenizerFast,
    GemmaConfig,
    Idefics2Config,
    Idefics2ForConditionalGeneration,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaTokenizerFast,
    MistralConfig,
    MistralForCausalLM,
    PaliGemmaConfig,
    PaliGemmaForConditionalGeneration,
    Phi3Config,
    Phi3ForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2Tokenizer,
    SiglipVisionConfig,
    T5Config,
    T5ForConditionalGeneration,
    T5TokenizerFast,
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


# Bart
tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")
vocab_size = tokenizer.vocab_size + len(tokenizer.added_tokens_encoder.keys())
config = BartConfig(
    vocab_size=tokenizer.vocab_size + len(tokenizer.added_tokens_encoder.keys()),
    d_model=16,
    d_kv=2,
    d_ff=64,
    num_layers=6,
    num_heads=8,
    decoder_start_token_id=0,
    is_encoder_decoder=False,
)
model = BartForCausalLM(config)
push_to_hub(model, tokenizer)


# Cohere
tokenizer = CohereTokenizerFast.from_pretrained("CohereForAI/aya-expanse-8b")
config = CohereConfig(
    vocab_size=tokenizer.vocab_size + len(tokenizer.added_tokens_encoder.keys()),
    hidden_size=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    num_hidden_layers=2,
    intermediate_size=32,
)
model = CohereForCausalLM(config)
push_to_hub(model, tokenizer)


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


# Llama 3
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
config = LlamaConfig(
    vocab_size=tokenizer.vocab_size + len(tokenizer.added_tokens_encoder.keys()),
    hidden_size=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    num_hidden_layers=2,
    intermediate_size=32,
)
model = LlamaForCausalLM(config)
push_to_hub(model, tokenizer, suffix="3")


# Llama 3.1
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
config = LlamaConfig(
    vocab_size=tokenizer.vocab_size + len(tokenizer.added_tokens_encoder.keys()),
    hidden_size=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    num_hidden_layers=2,
    intermediate_size=32,
)
model = LlamaForCausalLM(config)
push_to_hub(model, tokenizer, suffix="3.1")


# Llama 3.2
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
config = LlamaConfig(
    vocab_size=tokenizer.vocab_size + len(tokenizer.added_tokens_encoder.keys()),
    hidden_size=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    num_hidden_layers=2,
    intermediate_size=32,
)
model = LlamaForCausalLM(config)
push_to_hub(model, tokenizer, suffix="3.2")


# Mistral v0.1
tokenizer = LlamaTokenizerFast.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
config = MistralConfig(
    vocab_size=tokenizer.vocab_size + len(tokenizer.added_tokens_encoder.keys()),
    hidden_size=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    num_hidden_layers=2,
    intermediate_size=32,
)
model = MistralForCausalLM(config)
push_to_hub(model, tokenizer, suffix="0.1")


# Mistral v0.2
tokenizer = LlamaTokenizerFast.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
config = MistralConfig(
    vocab_size=tokenizer.vocab_size + len(tokenizer.added_tokens_encoder.keys()),
    hidden_size=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    num_hidden_layers=2,
    intermediate_size=32,
)
model = MistralForCausalLM(config)
push_to_hub(model, tokenizer, suffix="0.2")


# Mistral v0.3
tokenizer = LlamaTokenizerFast.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
config = MistralConfig(
    vocab_size=tokenizer.vocab_size + len(tokenizer.added_tokens_encoder.keys()),
    hidden_size=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    num_hidden_layers=2,
    intermediate_size=32,
)
model = MistralForCausalLM(config)
push_to_hub(model, tokenizer, suffix="0.3")


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


# Phi3
tokenizer = LlamaTokenizerFast.from_pretrained("microsoft/Phi-3.5-mini-instruct")
config = Phi3Config(
    vocab_size=tokenizer.vocab_size + len(tokenizer.added_tokens_encoder.keys()),
    hidden_size=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    num_hidden_layers=2,
    intermediate_size=32,
)
model = Phi3ForCausalLM(config)
push_to_hub(model, tokenizer)


# Qwen
tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")
config = Qwen2Config(
    vocab_size=tokenizer.vocab_size + len(tokenizer.added_tokens_encoder.keys()),
    hidden_size=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    num_hidden_layers=2,
    intermediate_size=32,
)
model = Qwen2ForCausalLM(config)
push_to_hub(model, tokenizer)


# T5
tokenizer = T5TokenizerFast.from_pretrained("google/flan-t5-small")
config = T5Config(
    vocab_size=tokenizer.vocab_size + len(tokenizer.added_tokens_encoder.keys()),
    d_model=16,
    d_kv=2,
    d_ff=64,
    num_layers=6,
    num_heads=8,
    decoder_start_token_id=0,
)
model = T5ForConditionalGeneration(config)
push_to_hub(model, tokenizer)
