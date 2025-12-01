# Testing systems with ARC-AGI

Run ARC-AGI tasks against multiple model adapters (OpenAI, Anthropic, Gemini, Fireworks, Grok, OpenRouter, X.AI, custom etc.) with built-in rate limiting, retries, and scoring.

## Quickstart
0) Clone this repo:
```bash
git clone https://github.com/arcprize/arc-agi-benchmarking.git
cd arc-agi-benchmarking
```

1) Install (installs all adapters + SDKs):
```bash
pip install .
```

2) Single-task dry run (no API keys) with the local `random-baseline` adapter:
```bash
python main.py \
  --data_dir data/sample/tasks \
  --config random-baseline \
  --task_id 66e6c45b \
  --save_submission_dir submissions/random-single \
  --log-level INFO
```

3) Run all bundled sample tasks with the random solver:
```bash
python cli/run_all.py \
  --task_list_file data/sample/task_lists/sample_tasks.txt \
  --data_dir data/sample/tasks \
  --model_configs "random-baseline" \
  --submissions-root submissions/random-baseline-sample \
  --log-level INFO
```

4) Score the outputs you just generated:
```bash
python src/arc_agi_benchmarking/scoring/scoring.py \
  --task_dir data/sample/tasks \
  --submission_dir submissions/random-baseline-sample \
  --results_dir results/random-baseline-sample
```

If using the random solver, expect all the attempts to be incorrect.

If you want to run real models, change the `model_configs` and add the corresponding API keys (see Data and Config sections below).

## Data

Rather than using the sample data in `data/sample/tasks/`, you can use the real ARC-AGI tasks from the following repositories:

* ARC-AGI-1 (2019): `git clone https://github.com/fchollet/ARC-AGI.git data/arc-agi`
* ARC-AGI-2 (2025): `git clone https://github.com/arcprize/ARC-AGI-2.git data/arc-agi`

## CLI parameters
- `--data_dir`: Folder containing ARC task `.json` files (e.g., `data/sample/tasks`).
- `--config` / `--model_configs`: Model config name(s) from `models.yml`. Single-task uses `--config`; batch uses `--model_configs` (comma-separated).
- `--save_submission_dir` / `--submissions-root`: Where to write outputs. Single-task writes directly to `--save_submission_dir`; batch uses `--submissions-root` and creates one folder per config.
- `--num_attempts`: How many attempts per test pair (per task).
- `--retry_attempts`: Internal retries within an attempt if the provider call fails.
- `--log-level`: `DEBUG|INFO|WARNING|ERROR|CRITICAL|NONE`.
- `--enable-metrics`: Toggle metrics collection (saved in `metrics_output/`).
- Scoring-specific:
  - `--submission_dir`: Where your run wrote outputs
  - `--results_dir` Where to write aggregated metrics/results

## Running models
For runs beyond the Quickstart:
- Batch (recommended): `python cli/run_all.py` with your task list, model configs, data dir, submissions root, attempts/retries, and log level. Uses asyncio, provider rate limiting, and tenacity retries; outputs land in `<submissions-root>/<config_name>`.
- Single task (debug): `python main.py` with a single `--config`, `--task_id`, and your data dir/save directory and log level.
See the CLI parameters section for flag details.

## Configuring models and providers
Tests are run based on model configs. Model configs hold the configuration (max output tokens, temperature, pricing etc.) for each test.

Model configs live in `src/arc_agi_benchmarking/models.yml`. Example:
  ```yaml
  - name: "gpt-4o-2024-11-20"   # config name you reference on the CLI; typically includes the reasoning level for clarity (e.g., "-basic", "-advanced")
    model_name: "gpt-4o-2024-11-20"  # provider’s actual model id
    provider: "openai"         # must match an adapter
    max_output_tokens: 4096    # optional; provider-specific
    temperature: 0.0           # optional; provider-specific
    pricing:
      date: "2024-11-20"
      input: 5.00              # USD per 1M input tokens
      output: 15.00            # USD per 1M output tokens
  ```
  - Standard fields: `name`, `model_name`, `provider`, `pricing` (`input`/`output` per 1M tokens, `date` for traceability).
  - Provider kwargs: any extra keys become `kwargs` and are passed directly to the SDK (e.g., `temperature`, `max_output_tokens`, `stream`, etc.).
- Rate limits live in `provider_config.yml` (`rate`, `period` per provider).
- Environment: set provider keys (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `HUGGING_FACE_API_KEY`). Copy `.env.example` to `.env` and fill in.

### Testing a new model

1. Add a new model config: add an entry to `models.yml` with an existing provider; then use `--config <name>` on the CLI


2. If you're adding a new adapter:
    1. Create `src/arc_agi_benchmarking/adapters/<provider>.py` implementing `ProviderAdapter`
    2. Export it from `src/arc_agi_benchmarking/adapters/__init__.py`
    3. Add a branch in `main.py` (and any factories) so the provider name is recognized
    4. Add a config entry in `models.yml` pointing to `provider: "<provider>"`
    5. [Optional] Add tests (adapters and parsing) to cover basic flows

## Scoring

To score a run you'll need 1) your test's submission directory and 2) the source taskset (which contains the solutions)

Score a run:  

```bash
python src/arc_agi_benchmarking/scoring/scoring.py
  --task_dir <data_dir>/data/evaluation
  --submission_dir submissions/<config>
  --results_dir results/<config>
```

## Contributing and testing
- Add new providers/models in `src/arc_agi_benchmarking/adapters` and `models.yml`.
- Run tests: `pytest`.
- Use the bundled sample task + submission for quick scoring checks.
