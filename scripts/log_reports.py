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

import json
import logging
import os
from pathlib import Path

from tabulate import tabulate


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def process_log_file(log):
    failed_tests = []
    passed_tests = []

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
                            failed_tests.append([test_name, duration, log.stem.split("_")[0]])
                        else:
                            passed_tests.append([test_name, duration, log.stem.split("_")[0]])
                except json.JSONDecodeError as e:
                    logging.warning(f"Could not decode line in {log}: {e}")

    except FileNotFoundError as e:
        logging.error(f"Log file {log} not found: {e}")
    except Exception as e:
        logging.error(f"Error processing log file {log}: {e}")

    return failed_tests, passed_tests


def main():
    print(f"## 🤗 Results of the {os.environ['TEST_TYPE']} TRL tests.")

    log_files = list(Path().glob("*.log"))
    if not log_files:
        print("⚠️ No log file found! The tests did not run, check the GitHub action job.")
        return

    for log in log_files:
        failed, passed = process_log_file(log)

        if failed:
            print(f"### ❌ {len(failed)} failed test(s) in `{log}`")
            failed_table = [test[0].split("::")[:2] + [test[0].split("::")[-1][:30] + ".."] for test in failed]
            table = tabulate(failed_table, headers=["File", "Class", "Test Name"], tablefmt="grid")
            print(f"\n```\n{table}\n```\n")
        elif passed:
            print(f"### ✅ No failures in `{log}`")
        else:
            print(f"⚠️ Empty log file `{log}`! Check the GitHub action job.")

        # Clean up log file
        try:
            os.remove(log)
        except OSError as e:
            logging.warning(f"Could not remove log file {log}: {e}")


if __name__ == "__main__":
    main()
