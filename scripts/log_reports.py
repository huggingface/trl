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
import argparse
import json
import os
from datetime import date
from pathlib import Path

from tabulate import tabulate


MAX_LEN_MESSAGE = 2900  # slack endpoint has a limit of 3001 characters

parser = argparse.ArgumentParser()
parser.add_argument("--slack_channel_name", default="trl-push-ci")


def main(slack_channel_name=None):
    failed = []
    passed = []

    group_info = []

    total_num_failed = 0
    empty_file = False or len(list(Path().glob("*.log"))) == 0

    total_empty_files = []

    for log in Path().glob("*.log"):
        section_num_failed = 0
        i = 0
        with open(log) as f:
            for line in f:
                line = json.loads(line)
                i += 1
                if line.get("nodeid", "") != "":
                    test = line["nodeid"]
                    if line.get("duration", None) is not None:
                        duration = f'{line["duration"]:.4f}'
                        if line.get("outcome", "") == "failed":
                            section_num_failed += 1
                            failed.append([test, duration, log.name.split("_")[0]])
                            total_num_failed += 1
                        else:
                            passed.append([test, duration, log.name.split("_")[0]])
            empty_file = i == 0
        group_info.append([str(log), section_num_failed, failed])
        total_empty_files.append(empty_file)
        os.remove(log)
        failed = []
    no_error_payload = {
        "type": "section",
        "text": {
            "type": "plain_text",
            "text": "🌞 There were no failures!"
            if not any(total_empty_files)
            else "Something went wrong there is at least one empty file - please check GH action results.",
            "emoji": True,
        },
    }

    message = ""
    payload = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "🤗 Results of the {} TRL tests.".format(os.environ.get("TEST_TYPE", "")),
            },
        },
    ]
    if total_num_failed > 0:
        for i, (name, num_failed, failed_tests) in enumerate(group_info):
            if num_failed > 0:
                if num_failed == 1:
                    message += f"*{name}: {num_failed} failed test*\n"
                else:
                    message += f"*{name}: {num_failed} failed tests*\n"
                failed_table = []
                for test in failed_tests:
                    failed_report = test[0].split("::")
                    # Truncate the last string as some test names might be long
                    failed_report[-1] = failed_report[-1][:30] + ".."
                    failed_table.append(failed_report)
                failed_table = tabulate(
                    failed_table,
                    headers=["Test Location", "Test Case", "Test Name"],
                    showindex="always",
                    tablefmt="grid",
                    maxcolwidths=[12, 12, 12],
                )
                message += "\n```\n" + failed_table + "\n```"

            if total_empty_files[i]:
                message += f"\n*{name}: Warning! Empty file - please check the GitHub action job *\n"
        print(f"### {message}")
    else:
        payload.append(no_error_payload)

    if os.environ.get("TEST_TYPE", "") != "":
        from slack_sdk import WebClient

        if len(message) > MAX_LEN_MESSAGE:
            message = f"There are {total_num_failed} failed tests in total ! Cannot display the entire summary - please check the action results directly"

        if len(message) != 0:
            md_report = {
                "type": "section",
                "text": {"type": "mrkdwn", "text": message},
            }
            payload.append(md_report)
            action_button = {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "*For more details:*"},
                "accessory": {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Check Action results", "emoji": True},
                    "url": f"https://github.com/huggingface/trl/actions/runs/{os.environ['GITHUB_RUN_ID']}",
                },
            }
            payload.append(action_button)

        date_report = {
            "type": "context",
            "elements": [
                {
                    "type": "plain_text",
                    "text": f"On Push main {os.environ.get('TEST_TYPE')} test results for {date.today()}",
                },
            ],
        }
        payload.append(date_report)

        print(payload)

        client = WebClient(token=os.environ.get("SLACK_API_TOKEN"))
        client.chat_postMessage(channel=f"#{slack_channel_name}", text=message, blocks=payload)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.slack_channel_name)
