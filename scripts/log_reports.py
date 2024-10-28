import argparse
import json
import logging
import os
from datetime import date
from pathlib import Path

from slack_sdk import WebClient
from tabulate import tabulate

MAX_LEN_MESSAGE = 3000  # Slack endpoint has a limit of 3001 characters

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", default=".", help="Directory containing the log files")
parser.add_argument("--slack_webhook_url", required=True, help="Slack webhook URL")
parser.add_argument("--slack_channel_name", default="trl-push-ci", help="Slack channel name")
parser.add_argument("--max_message_length", default=MAX_LEN_MESSAGE, type=int, help="Maximum length of the Slack message")
parser.add_argument("--test_type", default=os.environ.get("TEST_TYPE", ""), help="Type of the test suite")

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def process_log_file(log):
    # Code to process log file and return failed_tests, passed_tests, section_num_failed
    # ...

def main(args):
    group_info = []
    total_num_failed = 0
    total_empty_files = []

    log_files = list(Path(args.log_dir).glob("*.log"))
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
            "text": {"type": "plain_text", "text": f"🤗 Results of the {args.test_type} TRL tests."},
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

        if len(message) > args.max_message_length:
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
                        "text": f"On Push main {args.test_type} results for {date.today()}",
                    }
                ],
            }
        )

        # Send to Slack
        slack_client = WebClient(url=args.slack_webhook_url)
        slack_client.chat_postMessage(channel=f"#{args.slack_channel_name}", text=message, blocks=payload)

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
    main(args)
