# TRL Project - Claude Code Commands

This directory contains project-specific Claude Code slash commands for the TRL repository.

## Available Commands

### `/resolve-issue`

Resolve GitHub issues by creating or updating pull requests with automatic tracking.

**Usage:**

Create PR from issue:
```
/resolve-issue issue:4376
/resolve-issue issue:https://github.com/huggingface/trl/issues/4376
```

Modify existing PR:
```
/resolve-issue pr:4421
/resolve-issue pr:https://github.com/huggingface/trl/pull/4421
```

### `/pr-list`

List and manage all your tracked pull requests.

**Usage:**

```bash
/pr-list                # Show all tracked PRs
/pr-list status:open    # Show only open PRs
/pr-list status:closed  # Show merged/closed PRs
/pr-list cleanup        # Archive merged PRs
```

**Features:**
- ✅ Real-time status from GitHub
- ✅ Quick branch switching
- ✅ Automatic archiving of merged PRs
- ✅ Summary table of all your work

**Features:**
- ✅ Automatic branch creation and management
- ✅ PR tracking in `.claude/pr-tracker.json`
- ✅ Works on correct branch for each PR
- ✅ Handles git remote URL management
- ✅ Verifies changes against source code
- ✅ Uses `gh` CLI for GitHub operations

**Workflow:**
1. Fetches and analyzes the GitHub issue
2. Creates a descriptive branch
3. Implements changes with verification
4. Commits with proper format (no AI mention)
5. Pushes to your fork
6. Creates PR to upstream repository
7. Tracks PR for future modifications

**Example Session:**

```bash
# First time - Create PR from issue
/resolve-issue issue:4376

# Later - Modify the same PR
/resolve-issue pr:4421
```

The command automatically:
- Switches to the correct branch
- Pulls latest changes
- Applies your modifications
- Pushes updates (PR auto-updates)

## PR Tracking

Active PRs are tracked in `.claude/pr-tracker.json`:

```json
{
  "issue-4376": {
    "issue_number": 4376,
    "pr_number": 4421,
    "pr_url": "https://github.com/huggingface/trl/pull/4421",
    "branch": "docs/update-peft-integration",
    "status": "open"
  }
}
```

This file is `.gitignore`d and stays local to your machine.

## Working with Multiple PRs

The PR tracker is designed to handle multiple concurrent PRs seamlessly.

### Example: Multiple PR Workflow

```bash
# Morning: Create PR for issue #4376
/resolve-issue issue:4376
# ✅ Creates branch: docs/update-peft-integration
# ✅ Creates PR #4421
# ✅ Tracked in pr-tracker.json

# Start another PR for issue #4380
/resolve-issue issue:4380
# ✅ Creates branch: feat/add-new-trainer
# ✅ Creates PR #4425
# ✅ Added to pr-tracker.json

# Check all your PRs
/pr-list
# Shows:
#   Issue #4376 → PR #4421 (docs/update-peft-integration) - OPEN
#   Issue #4380 → PR #4425 (feat/add-new-trainer) - OPEN

# Switch back to first PR
/resolve-issue pr:4421
# ✅ Auto-switches to docs/update-peft-integration branch
# Make changes, commit, push

# Switch to second PR
/resolve-issue pr:4425
# ✅ Auto-switches to feat/add-new-trainer branch
# Make changes, commit, push

# After PR #4421 is merged
/pr-list cleanup
# ✅ Archives merged PR #4421
# ✅ Keeps active PR #4425
```

### What Happens in pr-tracker.json

As you work on multiple issues, the tracker grows:

```json
{
  "issue-4376": {
    "pr_number": 4421,
    "branch": "docs/update-peft-integration",
    "status": "open"
  },
  "issue-4380": {
    "pr_number": 4425,
    "branch": "feat/add-new-trainer",
    "status": "open"
  },
  "issue-4385": {
    "pr_number": 4430,
    "branch": "fix/trainer-bug",
    "status": "open"
  }
}
```

### Key Benefits

✅ **No Branch Confusion**: Each PR has its own branch, automatically managed
✅ **Quick Switching**: `/resolve-issue pr:4421` switches to the right branch
✅ **Status Tracking**: `/pr-list` shows status of all PRs at once
✅ **Clean Workspace**: `cleanup` archives merged PRs automatically
✅ **Concurrent Work**: Work on 5, 10, or 20 PRs simultaneously

## Current PRs

- **PR #4421**: PEFT integration documentation rewrite
  - Issue: #4376
  - Branch: `docs/update-peft-integration`
  - Status: Open
  - URL: https://github.com/huggingface/trl/pull/4421

Run `/pr-list` to see all tracked PRs with current status.

## Git Configuration

The command handles git remote management automatically:

```bash
# Automatically updates remote to fork if needed
git remote set-url origin https://github.com/behroozazarkhalili/trl.git

# Creates PR to upstream
gh pr create --repo huggingface/trl --head behroozazarkhalili:<branch>
```

## Requirements

- `gh` CLI installed and authenticated (`gh auth login`)
- Git configured with your credentials
- Fork of the repository (behroozazarkhalili/trl)

## Tips

1. **Always verify source code**: The command uses Read/Grep to verify changes
2. **No AI mention**: Commits and PRs never mention AI assistance
3. **Branch naming**: Uses conventional prefixes (docs/, feat/, fix/)
4. **Issue linking**: Always includes "Resolves #<number>" in PR description
5. **Incremental work**: You can modify PRs multiple times safely

## File Structure

```
.claude/
├── commands/
│   └── resolve-issue.md          # Slash command definition
├── pr-tracker.json                # PR tracking (gitignored)
└── README.md                      # This file
```
