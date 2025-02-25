.PHONY: test precommit common_tests slow_tests test_examples tests_gpu

check_dirs := examples tests trl

ACCELERATE_CONFIG_PATH = `pwd`/examples/accelerate_configs
COMMAND_FILES_PATH = `pwd`/commands

test:
	python -m pytest -n auto --dist=loadfile -s -v --reruns 5 --reruns-delay 1 --only-rerun '(OSError|Timeout|HTTPError.*502|HTTPError.*504||not less than or equal to 0.01)' ./tests/

precommit:
	python scripts/add_copyrights.py
	pre-commit run --all-files

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
