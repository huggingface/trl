# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

import ast
import inspect
import random
import shutil
import subprocess
import tempfile
import textwrap
import unittest

import torch
from transformers import is_bitsandbytes_available, is_comet_available, is_sklearn_available, is_wandb_available
from transformers.testing_utils import torch_device
from transformers.utils import is_rich_available

from trl import BaseBinaryJudge, BasePairwiseJudge
from trl.import_utils import (
    is_diffusers_available,
    is_joblib_available,
    is_llm_blender_available,
    is_mergekit_available,
    is_vllm_available,
)


# transformers.testing_utils contains a require_bitsandbytes function, but relies on pytest markers which we don't use
# in our test suite. We therefore need to implement our own version of this function.
def require_bitsandbytes(test_case):
    """
    Decorator marking a test that requires bitsandbytes. Skips the test if bitsandbytes is not available.
    """
    return unittest.skipUnless(is_bitsandbytes_available(), "test requires bitsandbytes")(test_case)


def require_comet(test_case):
    """
    Decorator marking a test that requires Comet. Skips the test if Comet is not available.
    """
    return unittest.skipUnless(is_comet_available(), "test requires comet_ml")(test_case)


def require_diffusers(test_case):
    """
    Decorator marking a test that requires diffusers. Skips the test if diffusers is not available.
    """
    return unittest.skipUnless(is_diffusers_available(), "test requires diffusers")(test_case)


def require_llm_blender(test_case):
    """
    Decorator marking a test that requires llm-blender. Skips the test if llm-blender is not available.
    """
    return unittest.skipUnless(is_llm_blender_available(), "test requires llm-blender")(test_case)


def require_mergekit(test_case):
    """
    Decorator marking a test that requires mergekit. Skips the test if mergekit is not available.
    """
    return unittest.skipUnless(is_mergekit_available(), "test requires mergekit")(test_case)


def require_rich(test_case):
    """
    Decorator marking a test that requires rich. Skips the test if rich is not available.
    """
    return unittest.skipUnless(is_rich_available(), "test requires rich")(test_case)


def require_sklearn(test_case):
    """
    Decorator marking a test that requires sklearn. Skips the test if sklearn is not available.
    """
    return unittest.skipUnless(is_sklearn_available() and is_joblib_available(), "test requires sklearn")(test_case)


def require_vllm(test_case):
    """
    Decorator marking a test that requires vllm. Skips the test if vllm is not available.
    """
    return unittest.skipUnless(is_vllm_available(), "test requires vllm")(test_case)


def require_no_wandb(test_case):
    """
    Decorator marking a test that requires no wandb. Skips the test if wandb is available.
    """
    return unittest.skipUnless(not is_wandb_available(), "test requires no wandb")(test_case)


def require_3_accelerators(test_case):
    """
    Decorator marking a test that requires at least 3 accelerators. Skips the test if 3 accelerators are not available.
    """
    torch_accelerator_module = getattr(torch, torch_device, torch.cuda)
    return unittest.skipUnless(
        torch_accelerator_module.device_count() >= 3, f"test requires at least 3 {torch_device}s"
    )(test_case)


class RandomBinaryJudge(BaseBinaryJudge):
    """
    Random binary judge, for testing purposes.
    """

    def judge(self, prompts, completions, gold_completions=None, shuffle_order=True):
        return [random.choice([0, 1, -1]) for _ in range(len(prompts))]


class RandomPairwiseJudge(BasePairwiseJudge):
    """
    Random pairwise judge, for testing purposes.
    """

    def judge(self, prompts, completions, shuffle_order=True, return_scores=False):
        if not return_scores:
            return [random.randint(0, len(completion) - 1) for completion in completions]
        else:
            return [random.random() for _ in range(len(prompts))]


class TrlTestCase(unittest.TestCase):
    """
    Base test case for TRL tests. Sets up a temporary directory for testing.
    """

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()


def extract_imports_from_file(source_file):
    """Extract all non-relative imports from the caller's file, including conditional imports"""

    with open(source_file) as f:
        source_code = f.read()
        tree = ast.parse(source_code)

    imports = []
    source_lines = source_code.splitlines()

    def has_imports_in_node(node):
        """Check if a node contains any import statements"""
        for child in ast.walk(node):
            if isinstance(child, ast.Import):
                return True
            elif isinstance(child, ast.ImportFrom) and child.level == 0:
                return True
        return False

    def get_source_segment(node):
        """Get the source code for a given AST node"""
        start_line = node.lineno - 1
        end_line = node.end_lineno - 1 if hasattr(node, "end_lineno") and node.end_lineno else start_line

        lines = source_lines[start_line : end_line + 1]
        if lines:
            first_line = lines[0]
            indent = len(first_line) - len(first_line.lstrip())
            dedented_lines = []
            for line in lines:
                if line.strip():
                    dedented_lines.append(line[indent:] if len(line) > indent else line)
                else:
                    dedented_lines.append("")
            return "\n".join(dedented_lines)
        return ""

    # Walk through top-level nodes only
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            module = node.module or ""
            names = [alias.name for alias in node.names]
            imports.append(f"from {module} import {', '.join(names)}")
        elif isinstance(node, ast.If) and has_imports_in_node(node):
            conditional_code = get_source_segment(node)
            imports.append(conditional_code)

    return "\n".join(imports)


def distributed(func):
    def wrapper(tmp_path):
        # Extract the function source and create a script
        source = inspect.getsource(func)
        lines = source.split("\n")

        # Find the function definition line and replace it
        func_def_line = None
        for i, line in enumerate(lines):
            if line.strip().startswith("def ") and func.__name__ in line:
                func_def_line = i
                break

        if func_def_line is not None:
            # Replace the function name with 'main' and remove parameters
            indent = len(lines[func_def_line]) - len(lines[func_def_line].lstrip())
            lines[func_def_line] = " " * indent + "def main():"

        # Remove the first line (original function def) and dedent
        script_content = textwrap.dedent("\n".join(lines[1:]))

        # Get the file where the decorated function is defined
        func_file = inspect.getfile(func)
        imports = extract_imports_from_file(func_file)

        # Create the full script with main execution
        full_script = imports + "\n\n" + script_content + "\n\nif __name__ == '__main__':\n    main()"
        from pathlib import Path

        script_path = Path(".") / f"{func.__name__}.py"
        script_path.write_text(full_script)

        cmd = ["accelerate", "launch", str(script_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Test failed:\n{result.stderr}"

    return wrapper
