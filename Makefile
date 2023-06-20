.PHONY: test

check_dirs := examples tests trl

test:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/
