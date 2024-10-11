# Copyright 2023 The HuggingFace Team, the AllenNLP library authors. All rights reserved.
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
"""
Script to close stale issue. Taken in part from the AllenNLP repository.
https://github.com/allenai/allennlp.
"""

import os
from datetime import datetime as dt
from datetime import timezone

from github import Github


LABELS_TO_EXEMPT = [
    "good first issue",
    "good second issue",
    "feature request",
    "help wanted",
]


def main():
    g = Github(os.environ["GITHUB_TOKEN"])
    repo = g.get_repo("huggingface/trl")
    open_issues = repo.get_issues(state="open")

    for issue in open_issues:
        comments = sorted(issue.get_comments(), key=lambda i: i.created_at, reverse=True)
        involved_users = [comment.user.login for comment in comments]
        inactive_days = (dt.now(timezone.utc) - issue.updated_at).days
        is_old = (dt.now(timezone.utc) - issue.created_at).days >= 30
        has_comments = len([user for user in involved_users if user != "github-actions[bot]"]) > 0
        to_exempt = any(label.name.lower() in LABELS_TO_EXEMPT for label in issue.get_labels())

        if is_old and not to_exempt:
            if has_comments and inactive_days > 23:
                issue.create_comment(
                    "This issue has been automatically marked as stale because it has not had "
                    "recent activity. If you think this still needs to be addressed "
                    "please comment on this thread.\n\n"
                )
            elif involved_users and involved_users[0] == "github-actions[bot]" and inactive_days > 7:
                issue.edit(state="closed")


if __name__ == "__main__":
    main()
