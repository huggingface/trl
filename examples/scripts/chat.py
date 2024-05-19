# flake8: noqa
# Copyright 2024 The HuggingFace Team. All rights reserved.
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


from trl.commands.cli_utils import init_zero_verbose

init_zero_verbose()

import copy
import json
import os
import pwd
import re
import time
from threading import Thread

import torch
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from trl.commands.cli_utils import ChatArguments, TrlParser, init_zero_verbose
from trl.trainer.utils import get_quantization_config


HELP_STRING = """\

**TRL CHAT INTERFACE**

The chat interface is a simple tool to try out a chat model.

Besides talking to the model there are several commands:
- **clear**: clears the current conversation and start a new one
- **example {NAME}**: load example named `{NAME}` from the config and use it as the user input
- **set {SETTING_NAME}={SETTING_VALUE};**: change the system prompt or generation settings (multiple settings are separated by a ';').
- **reset**: same as clear but also resets the generation configs to defaults if they have been changed by **set**
- **save {SAVE_NAME} (optional)**: save the current chat and settings to file by default to `./chat_history/{MODEL_NAME}/chat_{DATETIME}.yaml` or `{SAVE_NAME}` if provided
- **exit**: closes the interface
"""

SUPPORTED_GENERATION_KWARGS = [
    "max_new_tokens",
    "do_sample",
    "num_beams",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
]

SETTING_RE = r"^set\s+[A-Za-z\s_]+=[A-Za-z\d\s.!\"#$%&'()*+,-/:<=>?@\[\]^_`{|}~]+(?:;\s*[A-Za-z\s_]+=[A-Za-z\d\s.!\"#$%&'()*+,-/:<=>?@\[\]^_`{|}~]+)*$"


class RichInterface:
    def __init__(self, model_name=None, user_name=None):
        self._console = Console()
        if model_name is None:
            self.model_name = "assistant"
        else:
            self.model_name = model_name
        if user_name is None:
            self.user_name = "user"
        else:
            self.user_name = user_name

    def stream_output(self, output_stream):
        """Stream output from a role."""
        # This method is originally from the FastChat CLI: https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/cli.py
        # Create a Live context for updating the console output
        text = ""
        self._console.print(f"[bold blue]<{self.model_name}>:")
        with Live(console=self._console, refresh_per_second=4) as live:
            # Read lines from the stream
            for i, outputs in enumerate(output_stream):
                if not outputs or i == 0:
                    continue
                text += outputs
                # Render the accumulated text as Markdown
                # NOTE: this is a workaround for the rendering "unstandard markdown"
                #  in rich. The chatbots output treat "\n" as a new line for
                #  better compatibility with real-world text. However, rendering
                #  in markdown would break the format. It is because standard markdown
                #  treat a single "\n" in normal text as a space.
                #  Our workaround is adding two spaces at the end of each line.
                #  This is not a perfect solution, as it would
                #  introduce trailing spaces (only) in code block, but it works well
                #  especially for console output, because in general the console does not
                #  care about trailing spaces.
                lines = []
                for line in text.splitlines():
                    lines.append(line)
                    if line.startswith("```"):
                        # Code block marker - do not add trailing spaces, as it would
                        #  break the syntax highlighting
                        lines.append("\n")
                    else:
                        lines.append("  \n")
                markdown = Markdown("".join(lines).strip(), code_theme="github-dark")
                # Update the Live console output
                live.update(markdown)
        self._console.print()
        return text

    def input(self):
        input = self._console.input(f"[bold red]<{self.user_name}>:\n")
        self._console.print()
        return input

    def clear(self):
        self._console.clear()

    def print_user_message(self, text):
        self._console.print(f"[bold red]<{self.user_name}>:[/ bold red]\n{text}")
        self._console.print()

    def print_green(self, text):
        self._console.print(f"[bold green]{text}")
        self._console.print()

    def print_red(self, text):
        self._console.print(f"[bold red]{text}")
        self._console.print()

    def print_help(self):
        self._console.print(Markdown(HELP_STRING))
        self._console.print()


def get_username():
    return pwd.getpwuid(os.getuid())[0]


def create_default_filename(model_name):
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    return f"{model_name}/chat_{time_str}.json"


def save_chat(chat, args, filename):
    output_dict = {}
    output_dict["settings"] = vars(args)
    output_dict["chat_history"] = chat

    folder = args.save_folder

    if filename is None:
        filename = create_default_filename(args.model_name_or_path)
        filename = os.path.join(folder, filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w") as f:
        json.dump(output_dict, f, indent=4)
    return os.path.abspath(filename)


def clear_chat_history(system_prompt):
    if system_prompt is None:
        chat = []
    else:
        chat = [{"role": "system", "content": system_prompt}]
    return chat


def parse_settings(user_input, current_args, interface):
    settings = user_input[4:].strip().split(";")
    settings = [(setting.split("=")[0], setting[len(setting.split("=")[0]) + 1 :]) for setting in settings]
    settings = dict(settings)
    error = False

    for name in settings:
        if hasattr(current_args, name):
            try:
                if isinstance(getattr(current_args, name), bool):
                    if settings[name] == "True":
                        settings[name] = True
                    elif settings[name] == "False":
                        settings[name] = False
                    else:
                        raise ValueError
                else:
                    settings[name] = type(getattr(current_args, name))(settings[name])
            except ValueError:
                interface.print_red(
                    f"Cannot cast setting {name} (={settings[name]}) to {type(getattr(current_args, name))}."
                )
        else:
            interface.print_red(f"There is no '{name}' setting.")

    if error:
        interface.print_red("There was an issue parsing the settings. No settings have been changed.")
        return current_args, False
    else:
        for name in settings:
            setattr(current_args, name, settings[name])
            interface.print_green(f"Set {name} to {settings[name]}.")

        time.sleep(1.5)  # so the user has time to read the changes
        return current_args, True


def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, revision=args.model_revision)

    torch_dtype = args.torch_dtype if args.torch_dtype in ["auto", None] else getattr(torch, args.torch_dtype)
    quantization_config = get_quantization_config(args)
    model_kwargs = dict(
        revision=args.model_revision,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map="auto",
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)

    if getattr(model, "hf_device_map", None) is None:
        model = model.to(args.device)

    return model, tokenizer


def parse_eos_tokens(tokenizer, eos_tokens, eos_token_ids):
    if tokenizer.pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    else:
        pad_token_id = tokenizer.pad_token_id

    all_eos_token_ids = []

    if eos_tokens is not None:
        all_eos_token_ids.extend(tokenizer.convert_tokens_to_ids(eos_tokens.split(",")))

    if eos_token_ids is not None:
        all_eos_token_ids.extend([int(token_id) for token_id in eos_token_ids.split(",")])

    if len(all_eos_token_ids) == 0:
        all_eos_token_ids.append(tokenizer.eos_token_id)

    return pad_token_id, all_eos_token_ids


def chat_cli():
    parser = TrlParser(ChatArguments)
    args = parser.parse_args_into_dataclasses()[0]
    if args.config == "default":
        args.config = os.path.join(os.path.dirname(__file__), "config/default_chat_config.yaml")
    if args.config.lower() == "none":
        args.config = None
    args = parser.update_dataclasses_with_config([args])[0]
    if args.examples is None:
        args.examples = {}

    current_args = copy.deepcopy(args)

    if args.user is None:
        user = get_username()
    else:
        user = args.user

    model, tokenizer = load_model_and_tokenizer(args)
    generation_streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

    pad_token_id, eos_token_ids = parse_eos_tokens(tokenizer, args.eos_tokens, args.eos_token_ids)

    interface = RichInterface(model_name=args.model_name_or_path, user_name=user)
    interface.clear()
    chat = clear_chat_history(current_args.system_prompt)
    while True:
        try:
            user_input = interface.input()

            if user_input == "clear":
                chat = clear_chat_history(current_args.system_prompt)
                interface.clear()
                continue

            if user_input == "help":
                interface.print_help()
                continue

            if user_input == "exit":
                break

            if user_input == "reset":
                interface.clear()
                current_args = copy.deepcopy(args)
                chat = clear_chat_history(current_args.system_prompt)
                continue

            if user_input.startswith("save") and len(user_input.split()) < 2:
                split_input = user_input.split()

                if len(split_input) == 2:
                    filename = split_input[1]
                else:
                    filename = None
                filename = save_chat(chat, current_args, filename)
                interface.print_green(f"Chat saved in {filename}!")
                continue

            if re.match(SETTING_RE, user_input):
                current_args, success = parse_settings(user_input, current_args, interface)
                if success:
                    chat = []
                    interface.clear()
                    continue

            if user_input.startswith("example") and len(user_input.split()) == 2:
                example_name = user_input.split()[1]
                if example_name in current_args.examples:
                    interface.clear()
                    chat = []
                    interface.print_user_message(current_args.examples[example_name]["text"])
                    user_input = current_args.examples[example_name]["text"]
                else:
                    interface.print_red(
                        f"Example {example_name} not found in list of available examples: {list(current_args.examples.keys())}."
                    )
                    continue

            chat.append({"role": "user", "content": user_input})

            generation_kwargs = dict(
                inputs=tokenizer.apply_chat_template(chat, return_tensors="pt", add_generation_prompt=True).to(
                    model.device
                ),
                streamer=generation_streamer,
                max_new_tokens=current_args.max_new_tokens,
                do_sample=current_args.do_sample,
                num_beams=current_args.num_beams,
                temperature=current_args.temperature,
                top_k=current_args.top_k,
                top_p=current_args.top_p,
                repetition_penalty=current_args.repetition_penalty,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_ids,
            )

            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            model_output = interface.stream_output(generation_streamer)
            thread.join()
            chat.append({"role": "assistant", "content": model_output})

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    chat_cli()
