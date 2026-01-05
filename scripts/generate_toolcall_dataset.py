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
from transformers.utils import get_json_schema


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
    # Fictitious functions to simulate tool calls
    def start_timer(duration: int) -> int:
        """
        Starts a timer for the specified duration in seconds.

        Args:
            duration: Duration in seconds to set the timer for.

        Returns:
            The duration set for the timer.
        """
        return duration

    def get_current_time(location: str) -> str:
        """
        Returns the current time in the specified location.

        Args:
            location: The location for which to get the current time.

        Returns:
            The current time in the specified location.
        """
        return "06:22:48"

    def get_air_quality_index(location: str) -> int:
        """
        Returns the air quality index for the specified location.

        Args:
            location: The location for which to get the air quality index.

        Returns:
            The air quality index for the specified location.
        """
        return 53

    def play_music(title: str, artist: str) -> dict:
        """
        Plays music by the specified title and artist.

        Args:
            title: The title of the music to play.
            artist: The artist of the music to play.

        Returns:
            A dictionary indicating the status of the music playback.
        """
        return {"status": "Playing"}

    def get_weather_forecast(city: str, date: str) -> dict:
        """
        Returns the weather forecast for the specified city and date.

        Args:
            city: The city for which to get the weather forecast.
            date: The date for which to get the weather forecast.

        Returns:
            A dictionary containing the temperature and weather condition.
        """
        return {"temperature": 22, "condition": "partly cloudy"}

    def control_light(room: str, state: str) -> dict:
        """
        Controls the light in the specified room.

        Args:
            room: The room where the light should be controlled.
            state: The desired state of the light ("on" or "off").

        Returns:
            A dictionary indicating the state of the light.
        """
        return {"state": state}

    def create_reminder(time: str, note: str) -> str:
        """
        Creates a reminder for the specified time and note.

        Args:
            time: The time for the reminder.
            note: The note for the reminder.

        Returns:
            A confirmation message indicating that the reminder has been set.
        """
        return "I'll remind you to call mom at 7 PM."

    def get_wind_conditions(city: str, unit: str) -> tuple[int, str]:
        """
        Returns the wind conditions for the specified city.

        Args:
            city: The city for which to get the wind conditions.
            unit: The unit of measurement for the wind speed (e.g., "mph").

        Returns:
            A tuple containing the wind speed and direction.
        """
        return 14, "NW"

    start_timer = get_json_schema(start_timer)
    get_current_time = get_json_schema(get_current_time)
    get_air_quality_index = get_json_schema(get_air_quality_index)
    play_music = get_json_schema(play_music)
    get_weather_forecast = get_json_schema(get_weather_forecast)
    control_light = get_json_schema(control_light)
    create_reminder = get_json_schema(create_reminder)
    get_wind_conditions = get_json_schema(get_wind_conditions)

    # fmt: off
    dataset = Dataset.from_dict({
        "messages": [
            [
                {"role": "user", "content": "Set a timer for 10 minutes."},
                {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "start_timer", "arguments": {"duration": 600}}}]},
                {"role": "tool", "name": "start_timer", "content": "600"},
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
                {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "get_air_quality", "arguments": {"location": "Lisbon, Portugal"}}}]},
                {"role": "tool", "name": "get_air_quality", "content": "53"},
                {"role": "assistant", "content": "The air quality is moderate."},
            ],
            [
                {"role": "user", "content": "Play some music."},
                {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "play_music", "arguments": {"title": "Take Five", "artist": "Dave Brubeck"}}}]},
                {"role": "tool", "name": "play_music", "content": "{'status': 'Playing'}"},
                {"role": "assistant", "content": "Enjoy the jazz tunes!"},
            ],
            [
                {"role": "user", "content": "What's the weather like tomorrow in Berlin?"},
                {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "get_weather_forecast", "arguments": {"city": "Berlin", "date": "2025-06-16"}}}]},
                {"role": "tool", "name": "get_weather_forecast", "content": "{'temperature': 22, 'condition': 'partly cloudy'}"},
                {"role": "assistant", "content": "Tomorrow in Berlin will be partly cloudy with a high of 22°C."}
            ],
            [
                {"role": "user", "content": "Turn on the living room lights."},
                {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "control_light", "arguments": {"room": "living room", "state": "on"}}}]},
                {"role": "tool", "name": "control_light", "content": "{'state': 'on'}"},
                {"role": "assistant", "content": "The living room lights are now on."}
            ],
            [
                {"role": "user", "content": "Remind me to call mom at 7 PM."},
                {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "create_reminder", "arguments": {"time": "19:00", "note": "Call mom"}}}]},
                {"role": "tool", "name": "create_reminder", "content": "Reminder set"},
                {"role": "assistant", "content": "Okay, I’ll remind you to call mom at 7 PM."}
            ],
            [
                {"role": "user", "content": "How strong is the wind in Chicago right now?"},
                {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "get_wind_conditions", "arguments": {"city": "Chicago", "unit": "mph"}}}]},
                {"role": "tool", "name": "get_wind_conditions", "content": "(14, 'NW')"},
                {"role": "assistant", "content": "The wind in Chicago is blowing at 14 mph from the northwest."}
            ]
        ],
        "tools": [
            [start_timer, create_reminder],
            [get_current_time],
            [get_air_quality_index, get_weather_forecast, get_wind_conditions],
            [play_music, control_light],
            [get_weather_forecast, get_wind_conditions],
            [control_light],
            [start_timer, create_reminder],
            [get_weather_forecast, get_wind_conditions],
        ]
    })
    dataset = dataset.train_test_split(test_size=test_size, shuffle=False)
    if push_to_hub:
        dataset.push_to_hub(repo_id)
    # fmt: on


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    main(script_args.test_size, script_args.push_to_hub, script_args.repo_id)
