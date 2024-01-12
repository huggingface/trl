.PHONY: test precommit benchmark_core benchmark_aux common_tests slow_tests_single_gpu slow_tests_multi_gpu run_sft_examples run_dpo_examples

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
	python -m pytest tests/test_sft_trainer.py $(if $(IS_GITHUB_CI),--report-log "common_sft.log",)
	python -m pytest tests/test_dpo_trainer.py $(if $(IS_GITHUB_CI),--report-log "common_dpo.log",)
	python -m pytest tests/test_ppo_trainer.py $(if $(IS_GITHUB_CI),--report-log "common_ppo.log",)
	python -m pytest tests/test_reward_trainer.py $(if $(IS_GITHUB_CI),--report-log "common_rm.log",)

slow_tests_single_gpu:
	CUDA_VISIBLE_DEVICES=0 python -m pytest tests/slow/test_sft_slow.py $(if $(IS_GITHUB_CI),--report-log "sft_slow_single.log",)
	CUDA_VISIBLE_DEVICES=0 python -m pytest tests/slow/test_dpo_slow.py $(if $(IS_GITHUB_CI),--report-log "dpo_slow_single.log",)

slow_tests_multi_gpu:
	CUDA_VISIBLE_DEVICES=0,1 python -m pytest tests/slow/test_sft_slow.py $(if $(IS_GITHUB_CI),--report-log "sft_slow_multi.log",)
	CUDA_VISIBLE_DEVICES=0,1 python -m pytest tests/slow/test_dpo_slow.py $(if $(IS_GITHUB_CI),--report-log "dpo_slow_multi.log",)

run_sft_examples:
	touch temp_results_sft_tests.txt
	for file in $(ACCELERATE_CONFIG_PATH)/*.yaml; do \
		TRL_ACCELERATE_CONFIG=$${file} bash $(COMMAND_FILES_PATH)/run_sft.sh; \
		echo $$?','$${file} >> temp_results_sft_tests.txt; \
	done

run_dpo_examples:
	touch temp_results_dpo_tests.txt
	for file in $(ACCELERATE_CONFIG_PATH)/*.yaml; do \
		TRL_ACCELERATE_CONFIG=$${file} bash $(COMMAND_FILES_PATH)/run_dpo.sh; \
		echo $$?','$${file} >> temp_results_dpo_tests.txt; \
	done