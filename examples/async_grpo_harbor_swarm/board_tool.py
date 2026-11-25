#!/usr/bin/env python3
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

"""Small dispatcher for Flock's message_board tools; arguments and results stay unchanged."""

import argparse
import json
import sys

from message_board import TOOLS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tool", choices=[f"message_board.{name}" for name in TOOLS])
    parser.add_argument("--json", default="{}")
    args = parser.parse_args()
    kwargs = json.load(sys.stdin) if args.json == "-" else json.loads(args.json)
    print(json.dumps(TOOLS[args.tool.split(".", 1)[1]](**kwargs)))


if __name__ == "__main__":
    main()
