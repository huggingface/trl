# List and Manage Tracked Pull Requests

Display and manage all tracked pull requests in the current project.

## Usage

```bash
# List all tracked PRs
/pr-list

# List only open PRs
/pr-list status:open

# List only merged/closed PRs
/pr-list status:closed

# Clean up merged PRs from tracker
/pr-list cleanup
```

## Workflow

### Display All PRs

1. **Read Tracker File**
   - Load `.claude/pr-tracker.json`
   - Parse all PR entries

2. **Fetch Current Status**
   - For each PR, run: `gh pr view <number> --json state,title,updatedAt`
   - Update status in memory (don't modify file yet)

3. **Display Table**
   - Show formatted table with:
     - Issue #
     - PR #
     - Branch
     - Status (open/merged/closed)
     - Title/Description
     - Last Updated
     - PR URL

### Status Filtering

When `status:<filter>` is provided:
- `status:open` - Show only open PRs
- `status:closed` - Show merged or closed PRs
- `status:merged` - Show only merged PRs

### Cleanup Mode

When `cleanup` is provided:

1. **Identify Stale PRs**
   - Query GitHub for PR status
   - Find merged or closed PRs

2. **Archive Stale PRs**
   - Move to `.claude/pr-tracker-archive.json`
   - Keep metadata for reference
   - Remove from active tracker

3. **Confirm Changes**
   - Show what was archived
   - Display active PRs remaining

## Output Format

### Table Display

```
Active Pull Requests (3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Issue   PR      Branch                          Status    Updated
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#4376   #4421   docs/update-peft-integration    OPEN      2 hours ago
#4380   #4425   feat/add-new-trainer            OPEN      1 day ago
#4355   #4410   fix/memory-leak                 MERGED    3 days ago
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Quick Actions:
  /resolve-issue pr:4421  - Continue working on PR #4421
  /pr-list cleanup        - Archive merged/closed PRs
```

### Detailed View

For each PR, optionally show:
```
PR #4421 (Issue #4376) - OPEN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Title: docs: Rewrite PEFT integration guide
  Branch: docs/update-peft-integration
  Created: 2025-11-02 12:00:00
  Updated: 2025-11-02 14:30:00
  URL: https://github.com/huggingface/trl/pull/4421

  Description:
  Rewrite PEFT integration guide with comprehensive examples

  Status: ✅ Open, ready for review
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Tracker Structure for Multiple PRs

The `.claude/pr-tracker.json` file grows as you work on more issues:

```json
{
  "issue-4376": {
    "issue_number": 4376,
    "pr_number": 4421,
    "pr_url": "https://github.com/huggingface/trl/pull/4421",
    "branch": "docs/update-peft-integration",
    "status": "open",
    "description": "Rewrite PEFT integration guide"
  },
  "issue-4380": {
    "issue_number": 4380,
    "pr_number": 4425,
    "pr_url": "https://github.com/huggingface/trl/pull/4425",
    "branch": "feat/add-new-trainer",
    "status": "open",
    "description": "Add XYZ trainer implementation"
  },
  "issue-4355": {
    "issue_number": 4355,
    "pr_number": 4410,
    "pr_url": "https://github.com/huggingface/trl/pull/4410",
    "branch": "fix/memory-leak",
    "status": "merged",
    "description": "Fix memory leak in DPO trainer"
  }
}
```

## Key Management Features

### Automatic Status Updates

When you run `/pr-list`, the command:
1. Queries GitHub for current PR status
2. Updates the tracker file automatically
3. Shows accurate real-time status

### Branch Switching Helper

The output includes quick commands:
```bash
# Switch to any PR's branch quickly
/resolve-issue pr:4421  # Switches to docs/update-peft-integration
/resolve-issue pr:4425  # Switches to feat/add-new-trainer
```

### Archive System

Merged/closed PRs are archived to `.claude/pr-tracker-archive.json`:

```json
{
  "archived_at": "2025-11-03T10:00:00Z",
  "prs": {
    "issue-4355": {
      "issue_number": 4355,
      "pr_number": 4410,
      "status": "merged",
      "merged_at": "2025-11-02T18:00:00Z",
      "branch": "fix/memory-leak"
    }
  }
}
```

## Important Notes

1. **Concurrent Work**: You can work on multiple PRs simultaneously
2. **Branch Safety**: Each PR has its own branch - no conflicts
3. **Status Sync**: Status is synced with GitHub on each `/pr-list` call
4. **Archive**: Old PRs are archived automatically, not deleted
5. **Quick Switch**: Use `/resolve-issue pr:<number>` to switch between PRs

## Example Workflow

```bash
# Morning: See what's pending
/pr-list

# Work on PR #4421
/resolve-issue pr:4421
# Make changes, commit, push

# Switch to another PR
/resolve-issue pr:4425
# Make changes, commit, push

# End of day: Check status
/pr-list

# Weekly: Clean up merged PRs
/pr-list cleanup
```

## Benefits for Multiple PRs

✅ **Never lose track** of which branch belongs to which PR
✅ **Quick switching** between different work items
✅ **Status at a glance** for all your contributions
✅ **Clean workspace** with automatic archiving
✅ **Conflict-free** work on multiple issues simultaneously

## Integration with `/resolve-issue`

The two commands work together:

```bash
# Create new PRs
/resolve-issue issue:4376  # Creates PR, adds to tracker
/resolve-issue issue:4380  # Creates another PR, adds to tracker

# View all your PRs
/pr-list

# Continue work on any PR
/resolve-issue pr:4421  # Auto-switches to correct branch
/resolve-issue pr:4425  # Auto-switches to different branch

# Clean up when done
/pr-list cleanup
```

This creates a complete workflow for managing multiple concurrent contributions!
