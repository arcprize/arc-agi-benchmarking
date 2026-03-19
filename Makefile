.PHONY: help install test test-verbose run-sample run-batch run-benchmark clean score upload

help:
	@echo "Available commands:"
	@echo "  make install        - Install package in editable mode"
	@echo "  make test           - Run all tests"
	@echo "  make test-verbose   - Run tests with verbose output"
	@echo "  make run-sample     - Run random baseline on sample task"
	@echo "  make run-batch      - Run random baseline on all sample tasks"
	@echo "  make run-benchmark CONFIG=<name> DATA_SOURCE=<path>  - Run a full local benchmark"
	@echo "  make score CONFIGS=\"config1,config2\"  - Score configs across all datasets"
	@echo "  make upload CONFIG=<name> DATASET=<v1|v2>  - Upload submissions to Hugging Face"
	@echo "  make clean          - Remove generated files and caches"

install:
	pip install -e .

test:
	pytest -q

test-verbose:
	pytest -v

# Run a single sample task with random baseline (no API calls)
run-sample:
	python main.py \
		--data_dir data/sample/tasks \
		--config random-baseline \
		--task_id 66e6c45b \
		--save_submission_dir submissions/random-sample \
		--log-level INFO

# Run all sample tasks with random baseline
run-batch:
	python cli/run_all.py \
		--data_dir data/sample/tasks \
		--config random-baseline \
		--save_submission_dir submissions/random-batch \
		--log-level INFO

# Score submissions
score:
	python -m arc_agi_benchmarking.scoring.scoring \
		--task_dir data/sample/tasks \
		--submission_dir submissions/random-batch \
		--print_logs

# Upload PUBLIC submissions to Hugging Face (NEVER semiprivate or private)
# Usage: make upload CONFIG=minimax-m2.5 DATASET=v1
# Usage: make upload CONFIG=minimax-m2.5 DATASET=v2
DATASET ?=
upload:
ifndef CONFIG
	$(error CONFIG is required. Usage: make upload CONFIG=<model-config> DATASET=<v1|v2>)
endif
ifndef DATASET
	$(error DATASET is required (v1 or v2). Usage: make upload CONFIG=<model-config> DATASET=<v1|v2>)
endif
	python cli/submission_cli.py upload \
		submissions/$(CONFIG)/public-$(DATASET)/evaluation \
		--model-name $(CONFIG) \
		--task-set arc_agi_$(DATASET)_public_eval

# Upload PUBLIC submissions to Hugging Face (NEVER semiprivate or private)
# Usage: make upload CONFIG=minimax-m2.5 DATASET=v1
# Usage: make upload CONFIG=minimax-m2.5 DATASET=v2
DATASET ?=
upload:
ifndef CONFIG
	$(error CONFIG is required. Usage: make upload CONFIG=<model-config> DATASET=<v1|v2>)
endif
ifndef DATASET
	$(error DATASET is required (v1 or v2). Usage: make upload CONFIG=<model-config> DATASET=<v1|v2>)
endif
	python cli/submission_cli.py upload \
		submissions/$(CONFIG)/public-$(DATASET)/evaluation \
		--model-name $(CONFIG) \
		--task-set arc_agi_$(DATASET)_public_eval

clean:
	rm -rf __pycache__ .pytest_cache
	rm -rf src/arc_agi_benchmarking/__pycache__
	rm -rf src/arc_agi_benchmarking/**/__pycache__
	rm -rf submissions/random-sample submissions/random-batch
	rm -rf logs/random-baseline
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".checkpoints" -exec rm -rf {} + 2>/dev/null || true
