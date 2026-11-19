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

# ruff: noqa: T201

"""
Run TRL's chat-template and data-utils tests with a new model as the only model of the model-parametrized tests.

Copies `tests/` to a temporary directory, saves the model's tokenizer or processor there (small files only), rewrites
the model lists of the copies, and runs pytest on them. Run from the root of the TRL repo; the repo is not modified.

    python make_tests.py <model_id_or_dir> [--revision <sha>]
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from transformers import AutoConfig, AutoProcessor, AutoTokenizer


parser = argparse.ArgumentParser()
parser.add_argument("model", help="Hub model id or local directory")
parser.add_argument("--revision", default=None, help="Hub revision (commit sha) to pin")
args = parser.parse_args()

tmp = Path(tempfile.mkdtemp(prefix="trl_new_model_tests_"))
shutil.copytree("tests", tmp / "tests")
shutil.copy("pyproject.toml", tmp)  # pytest settings
config = AutoConfig.from_pretrained(args.model, revision=args.revision)
is_vlm = "vision_config" in config.sub_configs
# The tests route on the class name in the id: "ForCausalLM" loads a tokenizer, "ForConditionalGeneration" a processor
suffix = "ForConditionalGeneration" if is_vlm else "ForCausalLM"
path = tmp / f"newmodel-{config.architectures[0].split('For')[0]}{suffix}"
(AutoProcessor if is_vlm else AutoTokenizer).from_pretrained(args.model, revision=args.revision).save_pretrained(path)


def replace_list(src, anchor, items, occurrence=0):
    # Replace the list that follows the `occurrence`-th match of `anchor` (a parametrize argname line)
    start = src.index("[", [m.end() for m in re.finditer(re.escape(anchor), src)][occurrence])
    depth = 0
    for end in range(start, len(src)):
        depth += {"[": 1, "]": -1}.get(src[end], 0)
        if depth == 0:
            break
    return src[:start] + "[" + items + "]" + src[end + 1 :]


model = f'"{path}"'
text_model = "" if is_vlm else model
vlm_model = model if is_vlm else ""
src = (tmp / "tests/test_chat_template_utils.py").read_text()
src = replace_list(
    src, 'class TestAddResponseSchema:\n    @pytest.mark.parametrize(\n        "tokenizer_name",', text_model
)
src = replace_list(src, '@pytest.mark.parametrize(\n        "processor_name",\n', vlm_model)
src = replace_list(src, 'class TestSupportsToolCalling:\n    @pytest.mark.parametrize(\n        "model_id",', model)
src = replace_list(src, '@pytest.mark.parametrize(\n        "model_id",', "", occurrence=1)  # does not support tools
src = replace_list(src, '@pytest.mark.parametrize(\n    "tokenizer_name",\n', model)  # TestGetTrainingChatTemplate
src = replace_list(src, '@pytest.mark.parametrize(\n    "model_name",\n', model)  # TestParseResponse
(tmp / "tests/candidate_test_chat_template_utils.py").write_text(src)

src = (tmp / "tests/test_data_utils.py").read_text()
src = replace_list(src, "class TestApplyChatTemplate(TrlTestCase):\n    tokenizers =", model)
(tmp / "tests/candidate_test_data_utils.py").write_text(src)

# The copies import the TRL of this checkout
env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [os.environ.get("PYTHONPATH"), os.getcwd()]))}
command = [sys.executable, "-m", "pytest", "tests/candidate_test_chat_template_utils.py"]
command += ["tests/candidate_test_data_utils.py", "-k", "newmodel", "-rfs", "-q", "-p", "no:cacheprovider"]
print(f"tests copied to {tmp}", flush=True)
sys.exit(subprocess.run(command, cwd=tmp, env=env).returncode)
