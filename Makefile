.PHONY: test precommit benchmark_core benchmark_aux common_tests slow_tests test_examples tests_gpu

check_dirs := examples tests trl

ACCELERATE_CONFIG_PATH = `pwd`/examples/accelerate_configs
COMMAND_FILES_PATH = `pwd`/commands


dev:
	[ -L "$(pwd)/trl/commands/scripts" ] && unlink "$(pwd)/trl/commands/scripts" || true
	pip install -e ".[dev]"
	ln -s `pwd`/examples/scripts/ `pwd`/trl/commands

test:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/

precommit:
	pre-commit run --all-files

benchmark_core:
	bash ./benchmark/benchmark_core.sh

benchmark_aux:
	bash ./benchmark/benchmark_aux.sh

tests_gpu:
	python -m pytest tests/test_* $(if $(IS_GITHUB_CI),--report-log "common_tests.log",)

slow_tests:
	python -m pytest tests/slow/test_* $(if $(IS_GITHUB_CI),--report-log "slow_tests.log",)

test_examples:
	touch temp_results_sft_tests.txt
	for file in $(ACCELERATE_CONFIG_PATH)/*.yaml; do \
		TRL_ACCELERATE_CONFIG=$${file} bash $(COMMAND_FILES_PATH)/run_sft.sh; \
		echo $$?','$${file} >> temp_results_sft_tests.txt; \
	done

	touch temp_results_dpo_tests.txt
	for file in $(ACCELERATE_CONFIG_PATH)/*.yaml; do \
		TRL_ACCELERATE_CONFIG=$${file} bash $(COMMAND_FILES_PATH)/run_dpo.sh; \
		echo $$?','$${file} >> temp_results_dpo_tests.txt; \
	done