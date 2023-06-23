.PHONY: test precommit

check_dirs := examples tests trl

test:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/

precommit:
	pre-commit run --all-files
