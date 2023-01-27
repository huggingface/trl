.PHONY: quality style test

test:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/

quality:
	black --check --line-length 119 --target-version py38 tests trl
	isort --check-only tests trl
	flake8  tests trl

style:
	black --line-length 119 --target-version py38 tests trl examples setup.py
	isort tests trl