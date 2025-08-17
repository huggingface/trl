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
import json
import logging
import os
from datetime import date
from pathlib import Path

from tabulate import tabulate


MAX_LEN_MESSAGE = 2900  # Slack endpoint has a limit of 3001 characters

parser = argparse.ArgumentParser()
parser.add_argument("--slack_channel_name", default="trl-push-ci")

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def process_log_file(log):
    failed_tests = []
    passed_tests = []
    section_num_failed = 0

    try:
        with open(log) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    test_name = data.get("nodeid", "")
                    duration = f"{data['duration']:.4f}" if "duration" in data else "N/A"
                    outcome = data.get("outcome", "")

                    if test_name:
                        if outcome == "failed":
                            section_num_failed += 1
                            failed_tests.append([test_name, duration, log.stem.split("_")[0]])
                        else:
                            passed_tests.append([test_name, duration, log.stem.split("_")[0]])
                except json.JSONDecodeError as e:
                    logging.warning(f"Could not decode line in {log}: {e}")

    except FileNotFoundError as e:
        logging.error(f"Log file {log} not found: {e}")
    except Exception as e:
        logging.error(f"Error processing log file {log}: {e}")

    return failed_tests, passed_tests, section_num_failed


def main(slack_channel_name):
    group_info = []
    total_num_failed = 0
    total_empty_files = []

    log_files = list(Path().glob("*.log"))
    if not log_files:
        logging.info("No log files found.")
        return

    for log in log_files:
        failed, passed, section_num_failed = process_log_file(log)
        empty_file = not failed and not passed

        total_num_failed += section_num_failed
        total_empty_files.append(empty_file)
        group_info.append([str(log), section_num_failed, failed])

        # Clean up log file
        try:
            os.remove(log)
        except OSError as e:
            logging.warning(f"Could not remove log file {log}: {e}")

    # Prepare Slack message payload
    payload = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"🤗 Results of the {os.environ.get('TEST_TYPE', '')} TRL tests."},
        },
    ]

    if total_num_failed > 0:
        message = ""
        for name, num_failed, failed_tests in group_info:
            if num_failed > 0:
                message += f"*{name}: {num_failed} failed test(s)*\n"
                failed_table = [
                    test[0].split("::")[:2] + [test[0].split("::")[-1][:30] + ".."] for test in failed_tests
                ]
                message += (
                    "\n```\n"
                    + tabulate(failed_table, headers=["Test Location", "Test Name"], tablefmt="grid")
                    + "\n```\n"
                )

            if any(total_empty_files):
                message += f"\n*{name}: Warning! Empty file - check GitHub action job*\n"

        # Logging
        logging.info(f"Total failed tests: {total_num_failed}")
        print(f"### {message}")

        if len(message) > MAX_LEN_MESSAGE:
            message = (
                f"❌ There are {total_num_failed} failed tests in total! Please check the action results directly."
            )

        payload.append({"type": "section", "text": {"type": "mrkdwn", "text": message}})
        payload.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "*For more details:*"},
                "accessory": {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Check Action results"},
                    "url": f"https://github.com/huggingface/trl/actions/runs/{os.environ['GITHUB_RUN_ID']}",
                },
            }
        )
        payload.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "plain_text",
                        "text": f"On Push main {os.environ.get('TEST_TYPE')} results for {date.today()}",
                    }
                ],
            }
        )

        # Send to Slack
        from slack_sdk import WebClient

        slack_client = WebClient(token=os.environ.get("SLACK_API_TOKEN"))
        slack_client.chat_postMessage(channel=f"#{slack_channel_name}", text=message, blocks=payload)

    else:
        payload.append(
            {
                "type": "section",
                "text": {
                    "type": "plain_text",
                    "text": "✅ No failures! All tests passed successfully.",
                    "emoji": True,
                },
            }
        )
        logging.info("All tests passed. No errors detected.")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.slack_channel_name)
