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

import os
import subprocess
import sys

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import DPOConfig, DPOTrainer


MODEL_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
MIXED_EXAMPLE = {
    "prompt": "What color is the sky?",
    "chosen": [
        {"role": "user", "content": "What color is the sky?"},
        {"role": "assistant", "content": "It is blue."},
    ],
    "rejected": [
        {"role": "user", "content": "What color is the sky?"},
        {"role": "assistant", "content": "It is green."},
    ],
}


def test_is_conversational_is_independent_of_hash_seed():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = f"from trl import is_conversational\nexample = {MIXED_EXAMPLE!r}\nprint(is_conversational(example))\n"
    results = set()
    for hash_seed in ["0", "1", "3", "5"]:
        env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": repo_root}
        process = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
        results.add(process.stdout.strip())
    assert results == {"True"}


def test_dpo_prepare_dataset_does_not_add_eos_to_message_lists(tmp_path):
    user_message = {"role": "user", "content": "What color is the sky?"}
    # The plain-text `completion` column makes `is_conversational` hash-seed dependent before the fix: when the key
    # pointed at `completion`, DPOTrainer took the non-conversational path, called `add_eos`, and crashed on
    # `example["chosen"].endswith(eos_token)` because `chosen` is a list of messages.
    dataset = Dataset.from_dict(
        {
            "prompt": [[user_message]],
            "chosen": [[user_message, {"role": "assistant", "content": "It is blue."}]],
            "rejected": [[user_message, {"role": "assistant", "content": "It is green."}]],
            "completion": ["It is blue."],
        }
    )
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    trainer = DPOTrainer(
        model=model,
        args=DPOConfig(output_dir=str(tmp_path), report_to="none"),
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    assert "prompt_ids" in trainer.train_dataset.column_names
