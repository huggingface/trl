.PHONY: test precommit

check_dirs := examples tests trl

precommit:
	pre-commit run --all-files
