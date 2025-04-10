# Used with e.g. Aider + SWE-Bench training, we clone the repo and launch the agent N times, task with fixing, store the history, train with GRPO

import git
import shutil
import tempfile
from typing import List, Optional


def get_any_diff(repo_path: str, commit_id: str, gitignore: Optional[List[str]] = None) -> str:
    """
    Get any diffs to the repo at the given path from the given commit.
    If there are untracked files, we raise a warning and add them if
    they look like common code files.
    
    Args:
        repo_path: Path to the repo
        commit_id: Commit hash to diff from
        gitignore: List of patterns to ignore
        
    Returns:
        String of the diff
    """
    ...

def clone_repo_at_commit(repo_url: str, commit_id: str, target_dir: Optional[str] = None) -> str:
    """
    Fast clone method with no git history. Initializes an empty repo, adds the remote then fetches the specific commit and checks it out.
    
    Args:
        repo_url: Repository URL
        commit_id: Commit hash to checkout
        target_dir: Optional target directory. If None, creates a temporary directory.
        
    Returns:
        Path to the cloned repository
    """
    if target_dir is None:
        target_dir = tempfile.mkdtemp()
    
    repo = git.Repo.init(target_dir)
    origin = repo.create_remote('origin', repo_url)
    origin.fetch(commit_id, depth=1)
    repo.git.checkout(commit_id)

    return target_dir

def clean_repo_dir(repo_path: str):
    """Clean tempfolder"""
    # linux / macos / (windows?)
    assert repo_path.startswith("/tmp/") or repo_path.startswith("/var/folders/"), "For safety, repo_path must be a temporary directory"
    shutil.rmtree(repo_path)