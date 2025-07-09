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

import argparse
import logging
import os
from datetime import date

from tabulate import tabulate


MAX_LEN_MESSAGE = 2900  # slack endpoint has a limit of 3001 characters

parser = argparse.ArgumentParser()
parser.add_argument("--slack_channel_name", default="trl-push-examples-ci")
parser.add_argument("--text_file_name", required=True)


def main(text_file_name, slack_channel_name=None):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    message = ""

    if os.path.isfile(text_file_name):
        final_results = {}

        try:
            with open(text_file_name) as file:
                for line in file:
                    result, config_name = line.strip().split(",")
                    config_name = config_name.split("/")[-1].split(".yaml")[0]
                    final_results[config_name] = int(result)
        except Exception as e:
            logger.error(f"Error reading file {text_file_name}: {str(e)}")
            final_results = {}

        no_error_payload = {
            "type": "section",
            "text": {
                "type": "plain_text",
                "text": "🌞 There were no failures on the example tests!"
                if not len(final_results) == 0
                else "Something went wrong there is at least one empty file - please check GH action results.",
                "emoji": True,
            },
        }

        total_num_failed = sum(final_results.values())
    else:
        no_error_payload = {
            "type": "section",
            "text": {
                "type": "plain_text",
                "text": "❌ Something is wrong with the workflow please check ASAP!"
                "Something went wrong there is no text file being produced. Please check ASAP.",
                "emoji": True,
            },
        }

        total_num_failed = 0

    test_type_name = text_file_name.replace(".txt", "").replace("temp_results_", "").replace("_", " ").title()

    payload = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "🤗 Results of the {} TRL {} example tests.".format(
                    os.environ.get("TEST_TYPE", ""), test_type_name
                ),
            },
        },
    ]

    if total_num_failed > 0:
        message += f"{total_num_failed} failed tests for example tests!"

        for test_name, failed in final_results.items():
            failed_table = tabulate(
                [[test_name, "✅" if not failed else "❌"]],
                headers=["Test Name", "Status"],
                showindex="always",
                tablefmt="grid",
                maxcolwidths=[12],
            )
            message += "\n```\n" + failed_table + "\n```"

        print(f"### {message}")
    else:
        payload.append(no_error_payload)

    if os.environ.get("TEST_TYPE", "") != "":
        try:
            from slack_sdk import WebClient
        except ImportError:
            logger.error("slack_sdk is not installed. Please install it to use Slack integration.")
            return

        if len(message) > MAX_LEN_MESSAGE:
            print(f"Truncating long message from {len(message)} to {MAX_LEN_MESSAGE}")
            message = message[:MAX_LEN_MESSAGE] + "..."

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
                    "text": f"On Push - main {os.environ.get('TEST_TYPE')} test results for {date.today()}",
                },
            ],
        }
        payload.append(date_report)

        print(payload)

        try:
            client = WebClient(token=os.environ.get("SLACK_API_TOKEN"))
            response = client.chat_postMessage(channel=f"#{slack_channel_name}", text=message, blocks=payload)
            if response["ok"]:
                logger.info("Message sent successfully to Slack.")
            else:
                logger.error(f"Failed to send message to Slack: {response['error']}")
        except Exception as e:
            logger.error(f"Error sending message to Slack: {str(e)}")

    if __name__ == "__main__":
        args = parser.parse_args()
        main(args.text_file_name, args.slack_channel_name)
