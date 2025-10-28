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

import os
import subprocess
import sys
from datetime import datetime


COPYRIGHT_HEADER = f"""# Copyright 2020-{datetime.now().year} The HuggingFace Team. All rights reserved.
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
"""


def get_tracked_python_files():
    """Get a list of all tracked Python files using git."""
    try:
        # Get the list of all tracked files from Git
        result = subprocess.run(["git", "ls-files"], stdout=subprocess.PIPE, text=True, check=True)
        # Split the result by lines to get individual file paths
        files = result.stdout.splitlines()
        # Filter only Python files
        py_files = [f for f in files if f.endswith(".py")]
        return py_files
    except subprocess.CalledProcessError as e:
        print(f"Error fetching tracked files: {e}")
        return []


def check_and_add_copyright(file_path):
    """Check if the file contains a copyright notice, and add it if missing."""
    if not os.path.isfile(file_path):
        print(f"[SKIP] {file_path} does not exist.")
        return

    with open(file_path, encoding="utf-8") as f:
        content = f.readlines()

    # Check if the exact copyright header exists
    if "".join(content).startswith(COPYRIGHT_HEADER):
        return True

    # If no copyright notice was found, prepend the header
    print(f"[MODIFY] Adding copyright to {file_path}.")
    with open(file_path, "w", encoding="utf-8") as f:
        # Write the copyright header followed by the original content
        f.write(COPYRIGHT_HEADER + "\n" + "".join(content))
    return False


def main():
    """Main function to check and add copyright for all tracked Python files."""
    py_files = get_tracked_python_files()
    if not py_files:
        print("No Python files are tracked in the repository.")
        return

    print(f"Checking {len(py_files)} Python files for copyright notice...")

    have_copyright = [check_and_add_copyright(file_path) for file_path in py_files]
    if not all(have_copyright):
        print("❌ Some files were missing the required copyright and have been updated.")
        sys.exit(1)
    else:
        print("✅ All files have the required copyright.")
        sys.exit(0)


if __name__ == "__main__":
    main()
