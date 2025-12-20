.PHONY: test precommit common_tests slow_tests tests_gpu test_experimental

check_dirs := examples tests trl

ACCELERATE_CONFIG_PATH = `pwd`/examples/accelerate_configs

test:
	pytest -v -m "todo" -s -v tests/

precommit:
	python scripts/add_copyrights.py
	pre-commit run --all-files
	doc-builder style trl tests docs/source --max_len 119

slow_tests:
	pytest -m "slow" tests/ $(if $(IS_GITHUB_CI),--report-log "slow_tests.log",)

test_experimental:
	pytest -k "experimental" -n auto -s -v