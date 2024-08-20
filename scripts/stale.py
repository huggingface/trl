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
]


def main():
    g = Github(os.environ["GITHUB_TOKEN"])
    repo = g.get_repo("huggingface/trl")
    open_issues = repo.get_issues(state="open")

    for issue in open_issues:
        comments = sorted(issue.get_comments(), key=lambda i: i.created_at, reverse=True)
        involved_users = [comment.user.login for comment in comments if comment.user.login != "github-actions[bot]"]
        if (
            len(involved_users) >= 0  # somebody has commented
            and involved_users[0].user.login == "github-actions[bot]"  # last comment from this bot
            and (dt.now(timezone.utc) - issue.updated_at).days > 7  # stale for 7 days
            and (dt.now(timezone.utc) - issue.created_at).days >= 30  # created for 30 days
            and not any(label.name.lower() in LABELS_TO_EXEMPT for label in issue.get_labels())
        ):
            issue.edit(state="closed")
        elif (
            len(involved_users) >= 0  # somebody has commented
            and (dt.now(timezone.utc) - issue.updated_at).days > 23  # inactive for 23 days
            and (dt.now(timezone.utc) - issue.created_at).days >= 30  # created for 30 days
            and not any(label.name.lower() in LABELS_TO_EXEMPT for label in issue.get_labels())
        ):
            issue.create_comment(
                "This issue has been automatically marked as stale because it has not had "
                "recent activity. If you think this still needs to be addressed "
                "please comment on this thread.\n\n"
            )


if __name__ == "__main__":
    main()
