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

import re
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
SHARED_DIRS = {"accelerate_configs", "datasets"}


def test_examples_index_matches_folders():
    folders = {d.name for d in (REPO_ROOT / "examples").iterdir() if d.is_dir() and d.name not in SHARED_DIRS}
    overview = (REPO_ROOT / "docs" / "source" / "example_overview.md").read_text()
    rows = set(re.findall(r"\| \[`([a-z0-9_]+)`\]\(https://github\.com/huggingface/trl/tree/main/examples/", overview))
    missing_rows = folders - rows
    missing_folders = rows - folders
    assert not missing_rows, f"Example folders missing from the Index table in example_overview.md: {missing_rows}"
    assert not missing_folders, f"Index rows without a matching examples/ folder: {missing_folders}"
