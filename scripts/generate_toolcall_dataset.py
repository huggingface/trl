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

from dataclasses import dataclass, field

from datasets import Dataset
from transformers import HfArgumentParser


@dataclass
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        test_size (`float`, *optional*, defaults to `0.1`):
            Fraction of the dataset to include in the test split.
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether to push the dataset to the Hugging Face Hub.
        repo_id (`str`, *optional*, defaults to `"trl-internal-testing/zen"`):
            Hugging Face repository ID to push the dataset to.
    """

    test_size: float = field(
        default=0.1,
        metadata={"help": "Fraction of the dataset to include in the test split."},
    )
    push_to_hub: bool = field(
        default=False,
        metadata={"help": "Whether to push the dataset to the Hugging Face Hub."},
    )
    repo_id: str = field(
        default="trl-internal-testing/toolcall",
        metadata={"help": "Hugging Face repository ID to push the dataset to."},
    )


def main(test_size, push_to_hub, repo_id):
    # fmt: off
    dataset = Dataset.from_dict({
        "messages": [
            [
                {"role": "user", "content": "Set a timer for 10 minutes."},
                {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "start_timer", "arguments": {"duration": 600}}}]},
                {"role": "tool","name": "start_timer","content": "OK"},
                {"role": "assistant", "content": "Timer set for 10 minutes."},
            ],
            [
                {"role": "user", "content": "What time is it in Tokyo?"},
                {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "get_current_time", "arguments": {"location": "Tokyo"}}}]},
                {"role": "tool", "name": "get_current_time", "content": "06:22:48"},
                {"role": "assistant", "content": "The current time in Tokyo is 06:22 AM."},
            ],
            [
                {"role": "user", "content": "Is the air clean today in Lisbon?"},
                {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "get_air_quality_index", "arguments": {"location": "Lisbon, Portugal"}}}]},
                {"role": "tool", "name": "get_air_quality", "content": "53"},
                {"role": "assistant", "content": "The air quality is moderate."},
            ],
            [
                {"role": "user", "content": "Play some music."},
                {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "play_music", "arguments": {"title": "Take Five", "artist": "Dave Brubeck"}}}]},
                {"role": "tool", "name": "play_music", "content": "OK"},
                {"role": "assistant", "content": "Enjoy the jazz tunes!"},
            ],
            [
                {"role": "user","content": "What's the weather like tomorrow in Berlin?"},
                {"role": "assistant","tool_calls": [{"type": "function","function": {"name": "get_weather_forecast","arguments": {"city": "Berlin", "date": "2025-06-16"}}}]},
                {"role": "tool","name": "get_weather_forecast","content": "Partly cloudy, 22°C"},
                {"role": "assistant","content": "Tomorrow in Berlin will be partly cloudy with a high of 22°C."}
            ],
            [
                {"role": "user","content": "Turn on the living room lights."},
                {"role": "assistant","tool_calls": [{"type": "function","function": {"name": "control_light","arguments": {"room": "living room", "state": "on"}}}]},
                {"role": "tool","name": "control_light","content": "Lights turned on"},
                {"role": "assistant","content": "The living room lights are now on."}
            ],
            [
                {"role": "user","content": "Remind me to call mom at 7 PM."},
                {"role": "assistant","tool_calls": [{"type": "function","function": {"name": "create_reminder","arguments": {"time": "19:00", "note": "Call mom"}}}]},
                {"role": "tool","name": "create_reminder","content": "Reminder set"},
                {"role": "assistant","content": "Okay, I’ll remind you to call mom at 7 PM."}
            ],
            [
                {"role": "user","content": "How strong is the wind in Chicago right now?"},
                {"role": "assistant","tool_calls": [{"type": "function","function": {"name": "get_wind_conditions", "arguments": {"city": "Chicago", "unit": "mph"}}}]},
                {"role": "tool","name": "get_wind_conditions","content": "14 mph, NW"},
                {"role": "assistant","content": "The wind in Chicago is blowing at 14 mph from the northwest."}
            ]
        ],
    })
    dataset = dataset.train_test_split(test_size=test_size, shuffle=False)
    if push_to_hub:
        dataset.push_to_hub(repo_id)
    # fmt: on


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    main(script_args.test_size, script_args.push_to_hub, script_args.repo_id)
