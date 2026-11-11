.PHONY: test precommit test_experimental claude clean-ai

# Transient infrastructure errors that are worth retrying, matched against "<ExceptionType>: <message>":
#  - OSError, Timeout, HTTPError 502/504: Hub flakiness
#  - out of memory, STATUS_ALLOC_FAILED: GPU memory pressure, matched on the message since the raising type varies
rerun_errors := (OSError|Timeout|HTTPError.*502|HTTPError.*504|out of memory|STATUS_ALLOC_FAILED)

# `--dist loadgroup` keeps the vLLM server tests (`xdist_group("vllm_server")`) on one worker, since they share a port
test:
	pytest -n auto --dist loadgroup -s -v --reruns 5 --reruns-delay 1 --only-rerun '$(rerun_errors)' tests

precommit:
	python scripts/add_copyrights.py
	pre-commit run --all-files

test_experimental:
	pytest -n auto -s -v tests/experimental

claude:
	mkdir -p .claude
	rm -rf .claude/skills
	ln -snf ../.agents/skills .claude/skills

# The `.agents/skills` line removes a leftover symlink from the old `make codex` setup; the tracked
# directory is left alone.
clean-ai:
	[ -L .agents/skills ] && rm .agents/skills || true
	rm -rf .claude/skills
