# Copyright 2025 The HuggingFace Team. All rights reserved.
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


import json
import re


# Extract the tool call from the response
def extract_tool_calls(response):
    """
    Extracts tool call information from a given response string.

    This function uses regular expressions to find all occurrences of the
    `<tool_call>` tag and extracts the content within those tags.

    Args:
        response (str): The string containing the response, possibly with tool call information.

    Returns:
        list: A list of strings, where each string is the content of a tool call.
              Returns an empty list if no tool calls are found.
    """
    pattern = r"<tool_call>(.*?)</tool_call>"
    matches = re.findall(pattern, response, re.DOTALL)
    return matches if matches else []


def execute_tool_call(tool_call_str):
    """
    Executes a tool call based on the provided JSON string.

    This function parses the JSON string to extract the function name and arguments,
    then executes the corresponding function.

    Args:
        tool_call_str (str): A JSON string containing the tool call information,
                             including the function name and arguments.

    Returns:
        Any: The result of the executed function.

    Raises:
        ValueError: If the JSON is invalid, a required key is missing, or the function is not found.
    """
    try:
        # Parse the JSON string
        tool_call = json.loads(tool_call_str)
        
        # Get function name and arguments
        func_name = tool_call["name"]
        func_args = tool_call["arguments"]
        
        # Get the function from global namespace
        if func_name in globals():
            func = globals()[func_name]
            # Call function with unpacked arguments
            return func(**func_args)
        else:
            raise ValueError(f"Function {func_name} not found")
            
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON in tool call")
    except KeyError as e:
        raise ValueError(f"Missing required key in tool call: {e}")