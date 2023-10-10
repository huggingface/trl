.PHONY: test precommit benchmark_core benchmark_aux

check_dirs := examples tests trl

test:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/

precommit:
	pre-commit run --all-files

benchmark_core:
	bash ./benchmark/benchmark_core.sh

benchmark_aux:
	bash ./benchmark/benchmark_aux.sh
