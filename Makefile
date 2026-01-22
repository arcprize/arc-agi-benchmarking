.PHONY: help install test test-verbose run-sample run-batch clean score

help:
	@echo "Available commands:"
	@echo "  make install        - Install package in editable mode"
	@echo "  make test           - Run all tests"
	@echo "  make test-verbose   - Run tests with verbose output"
	@echo "  make run-sample     - Run random baseline on sample task"
	@echo "  make run-batch      - Run random baseline on all sample tasks"
	@echo "  make score          - Score submissions against ground truth"
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

clean:
	rm -rf __pycache__ .pytest_cache
	rm -rf src/arc_agi_benchmarking/__pycache__
	rm -rf src/arc_agi_benchmarking/**/__pycache__
	rm -rf submissions/random-sample submissions/random-batch
	rm -rf logs/random-baseline
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".checkpoints" -exec rm -rf {} + 2>/dev/null || true
