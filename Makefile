.PHONY: quality style test

check_dirs := examples tests trl

test:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/

quality:
	black --check --line-length 119 --target-version py38 $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs)

style:
	black --line-length 119 --target-version py38 $(check_dirs)
	isort $(check_dirs)