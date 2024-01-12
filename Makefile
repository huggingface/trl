.PHONY: test precommit benchmark_core benchmark_aux common_tests slow_sft_tests slow_dpotests run_sft_examples run_dpo_examples

check_dirs := examples tests trl

ACCELERATE_CONFIG_PATH = `pwd`/examples/accelerate_configs
COMMAND_FILES_PATH = `pwd`/commands

test:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/

precommit:
	pre-commit run --all-files

benchmark_core:
	bash ./benchmark/benchmark_core.sh

benchmark_aux:
	bash ./benchmark/benchmark_aux.sh

tests_common_gpu:
	python -m pytest tests/test_* $(if $(IS_GITHUB_CI),--report-log "common_tests.log",)

slow_sft_tests:
	python -m pytest tests/slow/test_sft_slow.py $(if $(IS_GITHUB_CI),--report-log "sft_slow.log",)

slow_dpo_tests:
	python -m pytest tests/slow/test_dpo_slow.py $(if $(IS_GITHUB_CI),--report-log "dpo_slow.log",)

run_sft_examples:
	touch temp_results_sft_tests.txt
	for file in $(ACCELERATE_CONFIG_PATH)/*.yaml; do \
		CUDA_LAUNCH_BLOCKING=1 TRL_ACCELERATE_CONFIG=$${file} bash $(COMMAND_FILES_PATH)/run_sft.sh; \
		echo $$?','$${file} >> temp_results_sft_tests.txt; \
	done

run_dpo_examples:
	touch temp_results_dpo_tests.txt
	for file in $(ACCELERATE_CONFIG_PATH)/*.yaml; do \
		CUDA_LAUNCH_BLOCKING=1 TRL_ACCELERATE_CONFIG=$${file} bash $(COMMAND_FILES_PATH)/run_dpo.sh; \
		echo $$?','$${file} >> temp_results_dpo_tests.txt; \
	done