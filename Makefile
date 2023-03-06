.PHONY: quality style test

test:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/

quality:
	black --check --line-length 119 --target-version py38 tests trl examples
	isort --check-only tests trl examples
	flake8  tests trl examples

style:
	black --line-length 119 --target-version py38 tests trl examples setup.py
	isort tests trl