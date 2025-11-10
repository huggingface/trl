# Resolve GitHub Issue with PR Tracking

You are resolving a GitHub issue by creating or updating a pull request.

## Overview

This command automatically detects your repository access level and uses the appropriate workflow:

- **Direct Access Workflow**: If you have write access to the repository, branches are pushed directly to upstream
- **Fork Workflow**: If you don't have write access, branches are pushed to your fork and PR is created from fork

The command checks permissions using `gh api repos/{owner}/{repo} --jq .permissions.push` and handles everything automatically.

## Input Format

The user will provide either:
- `issue:<number>` - GitHub issue number (e.g., `issue:4376`)
- `issue:<url>` - Full GitHub issue URL
- `pr:<number>` - Existing PR number to modify (e.g., `pr:4421`)
- `pr:<url>` - Full GitHub PR URL to modify

### Optional Parameters

- `test-failures=true` - Check and fix test failures after PR is created/updated
  - If provided, will check PR CI status and investigate failing tests
  - If not provided, skips test failure checking (default behavior)

## Workflow

### For New Issues (Creating PR)

1. **Ensure Starting from Main Branch**
   - **CRITICAL**: Always start from local main branch
   - Check current branch: `git branch --show-current`
   - If not on main, switch to main: `git checkout main`
   - This ensures clean separation between different issues/PRs

2. **Fetch and Analyze Issue**
   - Use `gh issue view <number>` or WebFetch to get issue details
   - Analyze requirements and scope
   - Identify files that need changes

3. **Ensure Latest Code**
   - Fetch latest upstream: `git fetch upstream main`
   - Verify working with latest: `git log --oneline -1 upstream/main`
   - Reset if needed: `git reset --hard upstream/main`
   - This ensures you're working with the most recent codebase

4. **Check Repository Write Access**
   - **CRITICAL**: ALWAYS check write access at the start - NEVER assume or rely on memory
   - Extract repo owner/name from issue or git remote
   - **MANDATORY**: Run `gh api repos/{owner}/{repo}/collaborators/$(gh api user --jq '.login')/permission --jq '.permission'` before any work
   - **Direct Access** ("write" or "admin"): Can push branches directly to upstream repo
   - **Fork Access** ("read" or error): Must use fork workflow
   - Store decision: `has_write_access=true/false`
   - **IMPORTANT**: Do not prejudge or assume based on previous sessions

5. **Fix Git Credential Helper** (if misconfigured)
   - **Check for error**: If you see `git: 'credential-!gh' is not a git command`, the credential helper is misconfigured
   - **Fix it**:
     ```bash
     # Remove broken credential helper
     git config --local --unset credential.helper

     # Setup proper gh CLI authentication
     gh auth setup-git
     ```
   - **Verify**: Run `git config --get-all credential.helper` - should show `cache --timeout=3600` or similar
   - **Note**: This fix is only needed if the credential helper was previously misconfigured

6. **Configure Remote for Direct Push** (if has_write_access=true)
   - Set origin to upstream: `git remote set-url origin https://github.com/{owner}/{repo}.git`
   - **Note**: gh CLI authentication is already configured globally, no local config needed
   - **Key learning**: The gh CLI setup provides authentication automatically for all repos
   - If push still fails with 403, verify write access again or fall back to fork workflow

7. **Create Branch**
   - Branch naming: `docs/<short-description>`, `feat/<short-description>`, `fix/<short-description>`
   - Use descriptive but concise names
   - Run: `git checkout -b <branch-name>`
   - Note: Branch creation is same for both workflows

8. **Implement Changes**
   - Read existing files before modification
   - Verify changes against source code
   - Follow project conventions and patterns
   - Use ultrathink to ensure accuracy

9. **Verify Implementation**
   - Cross-check with actual source files
   - Verify examples match official scripts
   - Ensure consistency throughout changes
   - Run linters/tests if applicable
   - For code changes: verify commands/functions exist in codebase
   - For documentation: validate structure and test examples

9.5. **Code Quality Validation**

   **Validate all modified files in this session:**

   ```bash
   # Get list of all modified files that are staged
   MODIFIED_FILES=$(git diff --cached --name-only --diff-filter=ACMR)

   if [ -n "$MODIFIED_FILES" ]; then
     echo "🔍 Running code quality checks on modified files..."
     echo ""

     # Separate files by type
     PYTHON_FILES=$(echo "$MODIFIED_FILES" | grep '\.py$')
     MARKDOWN_FILES=$(echo "$MODIFIED_FILES" | grep '\.md$')
     OTHER_FILES=$(echo "$MODIFIED_FILES" | grep -v '\.\(py\|md\)$')

     # Validate Python files with ruff
     if [ -n "$PYTHON_FILES" ]; then
       echo "🐍 Python files:"
       echo "$PYTHON_FILES" | sed 's/^/  - /'
       echo ""

       # Step 1: Auto-format with ruff
       echo "📝 Formatting with ruff..."
       ruff format $PYTHON_FILES

       # Step 2: Check for code quality issues
       echo "✨ Checking code quality..."
       if ! ruff check $PYTHON_FILES; then
         echo ""
         echo "⚠️  Ruff found code quality issues."
         echo ""
         echo "Options:"
         echo "  1. Auto-fix: Run 'ruff check --fix' on these files"
         echo "  2. Manual review: Review issues and fix manually"
         echo "  3. Proceed anyway: Continue to commit (not recommended)"
         echo ""
         # Use AskUserQuestion tool to get user choice
         # If user chooses auto-fix, run:
         #   ruff check --fix $PYTHON_FILES
         #   git add $PYTHON_FILES
       else
         echo "✅ Python files: All quality checks passed!"
       fi

       # Re-stage formatted files
       git add $PYTHON_FILES
       echo ""
     fi

     # List Markdown files (no auto-formatting, just awareness)
     if [ -n "$MARKDOWN_FILES" ]; then
       echo "📝 Markdown files modified:"
       echo "$MARKDOWN_FILES" | sed 's/^/  - /'
       echo "ℹ️  Markdown files will be committed as-is"
       echo ""
     fi

     # List other files
     if [ -n "$OTHER_FILES" ]; then
       echo "📄 Other files modified:"
       echo "$OTHER_FILES" | sed 's/^/  - /'
       echo "ℹ️  Other files will be committed as-is"
       echo ""
     fi

     # Re-stage all modified files to capture any formatting changes
     echo "📦 Re-staging all modified files..."
     git add $MODIFIED_FILES
     echo ""
   else
     echo "ℹ️  No files modified, skipping validation"
   fi
   ```

   **Key Benefits:**
   - ✅ Validates ALL modified files in current session (Python, Markdown, etc.)
   - ✅ Automatic ruff formatting for Python files
   - ✅ Quality checks catch issues before commit
   - ✅ User sees complete list of modified files before approval
   - ✅ Non-blocking: allows proceeding even if checks fail
   - ✅ Re-stages all files after validation to capture changes

10. **Request User Approval for Commit**
   - Present summary of changes made
   - Show proposed commit message
   - **IMPORTANT**: Ask user to confirm before committing
   - Use AskUserQuestion tool to get approval
   - Only proceed after user confirms

11. **Commit Changes (After Approval)**
   - Clear, descriptive commit message
   - Reference issue number: "Resolves #<issue-number>"
   - **CRITICAL**: Do NOT mention AI assistance, Claude, or any AI tool as author or contributor
   - **FORBIDDEN**: No "Generated with Claude Code", "Co-Authored-By: Claude", or similar attribution
   - Format:
     ```
     <type>: <short description>

     Resolves #<issue-number>

     - Bullet point of change 1
     - Bullet point of change 2
     ```

12. **Check Branch Status and Update if Needed**
   - **IMPORTANT**: Before pushing, check if branch is out-of-date with base branch
   - Run: `gh pr view <pr-number> --json mergeable,mergeStateStatus` (if PR exists)
   - If message shows "This branch is out-of-date with the base branch":
     ```bash
     # Fetch latest from main
     git fetch origin main

     # Merge main into your branch
     git merge origin/main

     # Resolve any conflicts if they appear
     # Then commit the merge
     ```
   - This prevents the "Merge the latest changes from main into this branch" warning
   - Ensures PR is always up-to-date before pushing new changes

13. **Push Branch**

   **IF has_write_access=true (Direct Access):**
   - Check remote points to upstream: `git remote -v`
   - If not, set it: `git remote set-url origin https://github.com/{owner}/{repo}.git`
   - Push directly: `git push -u origin <branch-name>`

   **IF has_write_access=false (Fork Access):**
   - Check current remote: `git remote -v`
   - Update to fork: `git remote set-url origin https://github.com/behroozazarkhalili/<repo>.git`
   - Push to fork: `git push -u origin <branch-name>`

14. **Create Pull Request**

   **IF has_write_access=true (Direct Access):**
   - Use `gh pr create` without --head flag (branch is in same repo):
     ```bash
     gh pr create \
       --base main \
       --title "<type>: <description>" \
       --body "..."
     ```

   **IF has_write_access=false (Fork Access):**
   - Use `gh pr create` with --head flag (branch is in fork):
     ```bash
     gh pr create \
       --repo <upstream-org>/<repo> \
       --base main \
       --head behroozazarkhalili:<branch-name> \
       --title "<type>: <description>" \
       --body "..."
     ```

   - **Both workflows include**:
     - "Resolves #<issue-number>"
     - Summary of changes
     - Detailed bullet points
     - Testing/verification notes

15. **Track PR**
   - Add entry to `.claude/pr-tracker.json` (preserves existing entries):
     ```json
     {
       "issue-<number>": {
         "issue_number": <number>,
         "pr_number": <pr-number>,
         "pr_url": "<url>",
         "branch": "<branch-name>",
         "workflow": "direct",
         "has_write_access": true,
         "created_at": "<timestamp>",
         "status": "open",
         "description": "<short-description>"
       }
     }
     ```
   - **Important**: Read existing file, add new entry, write back (don't overwrite!)
   - Each issue gets unique key: `issue-4376`, `issue-4380`, etc.
   - Multiple PRs coexist in same file
   - Include workflow type ("direct" or "fork") and has_write_access boolean

16. **Check and Fix Test Failures** (Only if `test-failures=true`)
   - **IMPORTANT**: This step is ONLY executed if user provides `test-failures=true`
   - If not provided, skip this step entirely and proceed to next step

   **IF test-failures=true:**
   - Wait a few minutes for CI checks to start running
   - Check PR CI status: `gh pr view <pr-number> --json statusCheckRollup`
   - Identify failing checks
   - For each failure:
     - Investigate the error (check logs, imports, syntax)
     - Fix the issue in the branch
     - Commit and push the fix
   - Repeat until all tests pass or no more fixable issues found

17. **Return to Main Branch**
   - **CRITICAL**: After PR is created and tracked (and test failures fixed if applicable), return to main branch
   - Run: `git checkout main`
   - This ensures clean state for next issue/task
   - Verify: `git branch --show-current` should show "main"

### For Existing PRs (Modifying)

1. **Load PR Context**
   - Check `.claude/pr-tracker.json` for PR details
   - Or use `gh pr view <number>` to get PR information
   - Extract branch name from PR

2. **Switch to PR Branch**
   - Run: `git checkout <branch-name>`
   - Verify: `git branch --show-current`
   - Pull latest: `git pull origin <branch-name>`

3. **Check if Branch is Out-of-Date**
   - **CRITICAL**: Always check before making changes
   - Run: `gh pr view <pr-number> --json mergeStateStatus --jq '.mergeStateStatus'`
   - If returns "BEHIND" or shows "out-of-date" warning:
     ```bash
     # Update branch with latest from main
     git fetch origin main
     git merge origin/main

     # Resolve conflicts if any
     # Push the merge
     git push origin <branch-name>
     ```
   - This ensures you're working on the latest code

4. **Make Modifications**
   - Read files before editing
   - Apply requested changes
   - Verify against source code

5. **Commit and Push**
   - Commit with descriptive message (NO AI attribution)
   - Push: `git push origin <branch-name>`
   - PR automatically updates

6. **Update Tracking**
   - Update `.claude/pr-tracker.json` with modification timestamp
   - Set status: "modified"

7. **Check and Fix Test Failures** (Only if `test-failures=true`)
   - **IMPORTANT**: This step is ONLY executed if user provides `test-failures=true`
   - If not provided, skip this step entirely and proceed to next step

   **IF test-failures=true:**
   - Wait a few minutes for CI checks to run
   - Check PR CI status: `gh pr view <pr-number> --json statusCheckRollup`
   - Identify failing checks
   - For each failure:
     - Investigate the error (check logs, imports, syntax)
     - Fix the issue in the branch
     - Commit and push the fix
   - Repeat until all tests pass or no more fixable issues found

8. **Return to Main Branch**
   - **CRITICAL**: After modifications are pushed (and test failures fixed if applicable), return to main branch
   - Run: `git checkout main`
   - This ensures clean state for next task
   - Verify: `git branch --show-current` should show "main"

## Important Rules

1. **CRITICAL: Always start from main branch** - Before starting any new issue/task, ensure you're on main branch. Check with `git branch --show-current`, switch if needed with `git checkout main`.
2. **CRITICAL: Always return to main branch** - After creating/modifying PR, always return to main branch with `git checkout main` to ensure clean state for next task.
3. **CRITICAL: Check write access FIRST** - MANDATORY: Run `gh api repos/{owner}/{repo}/collaborators/$(gh api user --jq '.login')/permission --jq '.permission'` at the very start of EVERY /resolve-issue invocation. NEVER assume, NEVER rely on memory, NEVER prejudge.
4. **Fix credential helper if broken** - If you see `git: 'credential-!gh' is not a git command`, run `git config --local --unset credential.helper` then `gh auth setup-git`
5. **Always verify information** against actual source code using Read/Grep tools
6. **CRITICAL: NEVER mention AI as author/contributor** - Absolutely FORBIDDEN in commits, PRs, documentation, or any git operations: "Generated with Claude Code", "Co-Authored-By: Claude", or any AI attribution. User has explicitly requested this restriction multiple times.
7. **CRITICAL: Always check if branch is out-of-date** - Before pushing changes to existing PRs, check if branch needs to be updated with latest from main. Use `gh pr view <number> --json mergeStateStatus` to check. If behind, merge main into branch first.
8. **Use gh CLI** for GitHub operations (preferred over MCP tools)
9. **Check authentication**: Run `gh auth status` if issues occur
10. **Track all PRs** in `.claude/pr-tracker.json` for future reference with correct workflow type
11. **Work on correct branch** - always verify with `git branch --show-current`
12. **Follow existing patterns** in the repository
13. **Direct access workflow**: If you have write access, push directly to upstream (simpler)
14. **Fork workflow**: Only use fork if you don't have write access
15. **No assumptions**: Each session is independent - always verify write access freshly

## Write Access Detection

**MANDATORY FIRST STEP - NO EXCEPTIONS**

Always check write access at the very beginning of EVERY /resolve-issue execution:

```bash
# Check if you have push access to the repository
gh api repos/huggingface/trl/collaborators/$(gh api user --jq '.login')/permission --jq '.permission'

# Returns:
#   "write" or "admin" -> You have write access (use direct workflow)
#   "read"             -> You don't have write access (use fork workflow)
#   error              -> You don't have write access (use fork workflow)
```

## Two Workflows

### Direct Access Workflow (has_write_access=true)

When you have write access to the upstream repository:

```bash
# 1. Ensure remote points to upstream
git remote set-url origin https://github.com/huggingface/trl.git

# 2. Create and switch to branch
git checkout -b docs/my-changes

# 3. Make changes, commit
git add .
git commit -m "docs: Update documentation"

# 4. Push directly to upstream
git push -u origin docs/my-changes

# 5. Create PR (no --head flag needed, branch is in same repo)
gh pr create \
  --base main \
  --title "docs: Update documentation" \
  --body "Description"
```

### Fork Workflow (has_write_access=false)

When you don't have write access (external contributors):

```bash
# 1. Ensure remote points to your fork
git remote set-url origin https://github.com/behroozazarkhalili/trl.git

# 2. Create and switch to branch
git checkout -b docs/my-changes

# 3. Make changes, commit
git add .
git commit -m "docs: Update documentation"

# 4. Push to your fork
git push -u origin docs/my-changes

# 5. Create PR (--head flag specifies fork)
gh pr create \
  --repo huggingface/trl \
  --base main \
  --head behroozazarkhalili:docs/my-changes \
  --title "docs: Update documentation" \
  --body "Description"
```

## Git Remote Management

If you get 403 errors when pushing:
```bash
# Check current remote
git remote -v

# Update based on write access:
#   Direct access: git remote set-url origin https://github.com/{owner}/{repo}.git
#   Fork access:   git remote set-url origin https://github.com/behroozazarkhalili/<repo>.git

# Verify
git remote -v

# Push
git push -u origin <branch-name>
```

## PR Tracker Format

Location: `.claude/pr-tracker.json`

### Single PR Example (Direct Access)
```json
{
  "issue-4376": {
    "issue_number": 4376,
    "issue_url": "https://github.com/huggingface/trl/issues/4376",
    "pr_number": 4421,
    "pr_url": "https://github.com/huggingface/trl/pull/4421",
    "branch": "docs/update-peft-integration",
    "repo": "huggingface/trl",
    "workflow": "direct",
    "has_write_access": true,
    "created_at": "2025-11-02T12:00:00Z",
    "modified_at": "2025-11-02T12:00:00Z",
    "status": "open",
    "description": "Rewrite PEFT integration guide"
  }
}
```

### Single PR Example (Fork Access)
```json
{
  "issue-4380": {
    "issue_number": 4380,
    "pr_number": 4425,
    "branch": "feat/add-new-trainer",
    "repo": "huggingface/trl",
    "fork": "behroozazarkhalili/trl",
    "workflow": "fork",
    "has_write_access": false,
    "status": "open"
  }
}
```

### Multiple PRs Example
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
  "issue-4390": {
    "pr_number": 4435,
    "branch": "docs/add-grpo-examples",
    "status": "open"
  }
}
```

### How Multiple PRs Work

1. **Each issue gets unique key**: `issue-4376`, `issue-4380`, etc.
2. **All PRs tracked together**: Work on 10+ PRs simultaneously
3. **Preserved on updates**: Adding new PR preserves existing entries
4. **Automatic cleanup**: Use `/pr-list cleanup` to archive merged PRs
5. **Branch switching**: `/resolve-issue pr:4421` finds correct branch from tracker

## Example Usage

Create new PR from issue:
```
/resolve-issue issue:4376
```

Create new PR and fix test failures:
```
/resolve-issue issue:4376 test-failures=true
```

Modify existing PR:
```
/resolve-issue pr:4421
```

Modify existing PR and fix test failures:
```
/resolve-issue pr:4421 test-failures=true
```

With full URL:
```
/resolve-issue issue:https://github.com/huggingface/trl/issues/4376
```

With full URL and test failure checking:
```
/resolve-issue issue:https://github.com/huggingface/trl/issues/4376 test-failures=true
```

## Success Criteria

- ✅ Started from main branch before creating new branch
- ✅ Write access checked and workflow determined
- ✅ Branch created with descriptive name
- ✅ Branch is up-to-date with main (merged latest if needed)
- ✅ All changes verified against source code
- ✅ All modified files validated (ruff for Python, listed for others)
- ✅ Commit message follows format (ABSOLUTELY NO AI/Claude attribution)
- ✅ Pushed successfully (direct to upstream OR to fork)
- ✅ PR created with proper description and workflow (NO AI attribution)
- ✅ PR tracked in `.claude/pr-tracker.json` with workflow type
- ✅ Issue referenced in PR description
- ✅ Zero mentions of AI, Claude, or automated tools in any git artifacts
- ✅ No "out-of-date with base branch" warnings on PR
- ✅ Test failures checked and fixed (only if `test-failures=true` was provided)
- ✅ Returned to main branch after completion

## Output

After completion, provide:
1. Workflow used (direct or fork)
2. PR URL
3. Branch name
4. Summary of changes
5. Next steps (if any)

Example output:
```
✅ PR Created Successfully!

Workflow: Direct Access (you have write access to huggingface/trl)
PR: https://github.com/huggingface/trl/pull/4421
Branch: docs/update-peft-integration
Status: Open, ready for review

Summary:
- Removed outdated conversational dataset conversion guidance
- Updated trainer documentation to reflect current capabilities
- Verified all trainers now support conversational data natively
```
