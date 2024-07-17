# Copyright 2022 The HuggingFace Team. All rights reserved.
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
#
# Function `strtobool` copied and adapted from `distutils` (as deprected
# in Python 3.10).
# Reference: https://github.com/python/cpython/blob/48f9d3e3faec5faaa4f7c9849fecd27eae4da213/Lib/distutils/util.py#L308-L321


def strtobool(val: str) -> bool:
    """Convert a string representation of truth to True or False booleans.

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.

    Raises:
        ValueError: if 'val' is anything else.
    """
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    if val in ("n", "no", "f", "false", "off", "0"):
        return False
    raise ValueError(f"Invalid truth value, it should be a string but {val} was provided instead.")
