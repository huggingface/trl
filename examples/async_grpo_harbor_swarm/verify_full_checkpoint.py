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

"""Reload the full engineering-smoke checkpoint; fail before native gates on bad saves."""

import json
import sys
from pathlib import Path

from transformers import AutoModelForImageTextToText, AutoTokenizer


root = Path(sys.argv[1])
result = json.loads((root / "training-smoke.json").read_text())
assert result["optimizer_steps"] == 2 and result["parameters_changed"] and result["mixed_groups"]
checkpoint = root / "checkpoint"
assert not (checkpoint / "adapter_config.json").exists(), "Expected full weights, not an adapter"
model = AutoModelForImageTextToText.from_pretrained(checkpoint, device_map="cpu", dtype="bfloat16")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
assert sum(parameter.numel() for parameter in model.parameters()) > 3_000_000_000
assert tokenizer.encode("checkpoint reload")
(root / "reload-check.json").write_text(json.dumps({"full_model_reload": True, **result}, indent=2))
print("Full model checkpoint reload passed", flush=True)
