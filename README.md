# Testing model baselines on ARC-AGI

This repo contains code for testing model baselines on ARC-AGI. The input data is a folder containing individual files for ARC-AGI tasks.


## Setup

`git clone https://github.com/arcprizeorg/model_baseline.git`

`git submodule update --init`

Install from the `pyproject.toml` (flit backend):

```bash
# Regular install (installs all adapters and their SDKs)
pip install .

# Editable/dev install
pip install -e .
```

We currently use an "install everything" strategy so imports work out of the box; this includes provider SDKs such as OpenAI, Anthropic, Google (gemini), Hugging Face Fireworks, etc. No `requirements.txt` is maintained.

## ARC-AGI-1 vs ARC-AGI-2

The task format for ARC-AGI-1 and ARC-AGI-2 are identical. You can point this testing harness towards ARC-AGI-2 via the `--data_dir` parameter. Ensure you\'re using the correct `<task_list>.txt` found in `data/task_lists/` for the set you\'re testing.

## Efficient Batch Testing with `cli/run_all.py` (Recommended)

The primary and recommended method for running multiple ARC tasks against various model configurations is the `cli/run_all.py` script. This script leverages `asyncio` for efficient concurrency within a single Python process, replacing the need for external tools like `GNU parallel`.

**Key Features & Benefits:**

*   **Concurrency:** Runs multiple (task, model_config) pairs concurrently using `asyncio`.
*   **Rate Limiting:** Implements proactive provider-level rate limiting. Configure limits (requests per period) in `provider_config.yml`.
*   **Resilient Retries:** Uses the `tenacity` library for automatic, exponential backoff retries on transient API errors (e.g., rate limit errors, server-side issues).
*   **Centralized Logging:** Consistent logging across the orchestrator and individual task solvers (`ARCTester`). Log verbosity is controlled via the `--log-level` argument (options: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`, `NONE`).
*   **Optional Metrics:** Performance timing metrics are **disabled by default**. They can be enabled using the `--enable-metrics` flag if detailed performance data is needed.
*   **Simplified Workflow:** Manages all aspects of a batch run, from task queuing to result aggregation.

**Environment Setup (API Keys):**

Before running the scripts, ensure you have a `.env` file in the root of the project. This file should contain the necessary API keys for the providers you intend to use. Copy the `.env.example` file to `.env` and fill in your API keys.

For example, if you are using OpenAI, your `.env` file should include:
`OPENAI_API_KEY="your_actual_openai_api_key"`

**Example Usage:**

```bash
# Run all tasks from public_evaluation_v1.txt against gpt-4o and claude_opus,
# with 2 attempts per task, INFO level logging, and metrics enabled.
python cli/run_all.py \
    --task_list_file data/task_lists/public_evaluation_v1.txt \
    --model_configs "gpt-4o-2024-11-20,claude_opus" \
    --num_attempts 2 \
    --log-level INFO \
    --enable-metrics

# Run a smaller set of tasks, with metrics disabled (default) and minimal logging
# Note: 'data/task_lists/sample_tasks.txt' is an example path.
# You will need to create this file yourself, for instance, by selecting a
# subset of task IDs from a larger list like 'public_evaluation_v1.txt'
# and saving them into 'data/task_lists/sample_tasks.txt' (one task ID per line).
python cli/run_all.py \
    --task_list_file data/task_lists/sample_tasks.txt \
    --model_configs "gpt-4o-2024-11-20" \
    --log-level WARNING
```

**Key CLI Arguments for `cli/run_all.py`:**

*   `--task_list_file`: Path to the `.txt` file containing task IDs (one per line). (Default: `data/task_lists/public_evaluation_v1.txt`)
*   `--model_configs`: Comma-separated list of model configuration names from `models.yml`. (Default: `gpt-4o-2024-11-20`)
*   `--data_dir`: Data set directory. (Default: `data/arc-agi/data/evaluation`)
*   `--submissions-root`: Root folder to save submissions. (Default: `submissions`)
*   `--overwrite_submission`: Overwrite existing submissions. (Default: `False`)
*   `--print_submission`: Enable `ARCTester` to log final submission content (at INFO level). (Default: `False`)
*   `--num_attempts`: Number of attempts by `ARCTester` for each prediction. (Default: `2`)
*   `--retry_attempts`: Number of internal retry attempts by `ARCTester` for failed predictions. (Default: `2`)
*   `--log-level`: Set logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`, `NONE`). (Default: `INFO`)
*   `--enable-metrics`: Enable metrics collection and dumping (disabled by default). (Default: `False`)

**Provider Rate Limit Configuration (`provider_config.yml`):**

Create a `provider_config.yml` file in the project root to specify API rate limits for different providers. If this file is not found, or a provider is not listed, a default rate limit (e.g., 400 requests / 60 seconds) will be used with a warning.

Example `provider_config.yml`:
```yaml
openai:
  rate: 5000    # Max requests
  period: 60    # Time period in seconds (e.g., 60 for per minute)
  # Calculated as 5000 requests per 60 seconds

anthropic:
  rate: 1000
  period: 60

gemini:
  rate: 60
  period: 60
# ... add other providers as needed
```

## Testing a Single Task (for Debugging or Detailed Analysis)

While `cli/run_all.py` is recommended for batch runs, you can still test a single task using `main.py`. This is useful for debugging specific tasks or adapter configurations.

```bash
python main.py --data_dir data/arc-agi/data/evaluation --config grok-4-0709 --task_id 0a1d4ef5 --log-level DEBUG --enable-metrics
```

### Verbose Mode for Debugging

For detailed debugging output, use the `--verbose` flag:

```bash
python main.py --data_dir data/arc-agi/data/evaluation --config grok-4-0709 --task_id 0a1d4ef5 --verbose
```

The `--verbose` flag provides:
- **DEBUG level logs** for `arc_agi_benchmarking` code (shows detailed execution flow)
- **Library noise reduction** - keeps external library loggers (OpenAI, Anthropic, httpx, etc.) at WARNING level
- **Response waiting indicators** - shows when the system is waiting for API responses
- **Full error tracebacks** - displays complete stack traces for debugging failures

This is particularly useful when:
- Debugging API connection issues
- Understanding token usage and costs
- Troubleshooting provider-specific problems
- Analyzing streaming behavior

Note: `main.py` also supports `--log-level` and `--enable-metrics` (metrics are disabled by default).

## Legacy Concurrency: Running with `GNU parallel` (Alternative)
Previously, `GNU parallel` was suggested for concurrency. While `cli/run_all.py` is now the preferred method due to its integrated features, you can still use `parallel` if needed, but it will not benefit from the built-in rate limiting, tenacity retries, or centralized logging of `cli/run_all.py`.

Example with `parallel`:
```bash
# brew install parallel # If not already installed
parallel --jobs 20 --progress python main.py --data_dir data/arc-agi/data/evaluation --config claude_sonnet --task_id {} --save_submission_dir submissions/claude_sonnet_parallel --log-level WARNING :::: ./data/task_lists/public_evaluation_v1.txt
```

To generate a task list for `parallel`:
`python src.utils.generate_tasks_list.py --task_dir data/arc-agi/data/training --output_file data/task_lists/public_training.txt`

## Scoring

You can score your submissions by pointing the scoring script at your submissions directory:

`python src.scoring.scoring.py --task_dir data/arc-agi/data/evaluation --submission_dir submissions/gpt-4o-2024-11-20 --results_dir results/gpt-4o-2024-11-20`
(Note: The old `--print_logs` for scoring might be replaced by standard logging; check scoring script.)

## Results

Results from the scoring script are stored in the `results` folder. Performance metrics from `cli/run_all.py` (if enabled) are saved in the `metrics_output` directory by default.

# Contributing

This repo welcomes contributions! Specifically, we would love help adding more model adapters to the `src/arc_agi_benchmarking/adapters` folder.

For more information visit the [ARC Prize](https://arcprize.org/).

## Contributing: Testing Providers

When implementing new providers or modifying existing ones, utilize `cli/run_all.py` for thorough testing across multiple tasks. This script provides a more robust testing environment than isolated single-task runs.
The `test_providers.sh` script was previously used for basic validation. While it might still offer some utility for quick checks, `cli/run_all.py` with a small task list is recommended for evaluating provider behavior under concurrency and rate limiting.

(The rest of the README, including sections on CLI Usage for Hugging Face, Adding New Providers/Models, and Running Tests (`pytest`), can remain largely as is, but ensure consistency with python/python3 and any new logging/metric flags if those scripts are also updated.)

### CLI Usage

#### Validation
Validate model outputs against task sets:
```bash
# Basic validation
python cli/main.py validate data/arc-agi/data/evaluation submissions/open_ai_o1_high_20241217

# Validate another model's outputs
python cli/main.py validate data/arc-agi/data/evaluation submissions/claude_sonnet_20241022
```

#### Upload
Upload a single model's outputs to a task set repository:
```bash
# Basic upload (private repository)
python cli/submission_cli.py upload submissions/open_ai_o1_high_20241217 --task-set public_eval_v1

# Upload to a different organization
python cli/submission_cli.py upload submissions/claude_sonnet_20241022 --task-set public_eval_v1 --org your-org-name

# Create a public repository
python cli/submission_cli.py upload <path to your submissions> --task-set arc_agi_v2_public_eval --model-name <model_name> --public
```

#### Bulk Upload
Upload multiple model outputs at once:
```bash
# Upload all models in submissions directory (private repository)
python cli/submission_cli.py bulk-upload submissions/ --task-set public_eval_v1

# Upload to a different organization
python cli/submission_cli.py bulk-upload submissions/ --task-set public_eval_v1 --org your-org-name

# Create a public repository
python cli/submission_cli.py bulk-upload submissions/ --task-set public_eval_v1 --public
```

Notes:
- All uploads create private repositories by default
- Use `--public` flag to create public repositories
- Files are uploaded to subdirectories matching model names
- Default organization is "arcprize"

### Hugging Face Upload

#### Authentication
Before uploading, you'll need to authenticate with Hugging Face:

1. Get your access token from https://huggingface.co/settings/tokens
2. Set up authentication using either method:
   ```bash
   # Option 1: Environment variable
   export HUGGING_FACE_HUB_TOKEN=your_token_here
   
   # Option 2: CLI login
   huggingface-cli login
   ```

#### Upload
The upload process organizes submissions by task sets. Each task set (e.g., public_eval_v1) becomes a separate dataset repository on Hugging Face, with model submissions organized in subdirectories.

Structure:
```
task_set_name/
├── model_name_1/
│   ├── result1.json
│   ├── result2.json
├── model_name_2/
│   ├── result1.json
│   └── result2.json
```

To upload model outputs:
```bash
python cli/submission_cli.py upload submissions/model_name --task-set task_set_name [--org organization] [--public] [--model-name model_name]
```

For example:
```bash
python cli/submission_cli.py upload submissions/open_ai_o1_high_20241217 --task-set public_eval_v1
```

#### Bulk Upload
To upload multiple model outputs at once:
```bash
python cli/submission_cli.py bulk-upload submissions/ --task-set task_set_name [--org organization] [--public]
```
## Contributing: Testing Providers

For contributors implementing new providers, we provide a streamlined way to validate your implementation using the `test_providers.sh` script. This script helps ensure your provider implementation works correctly with the ARC-AGI tasks before submitting a pull request.

### Running Provider Tests for Development

```bash
# Run all provider tests
./test_providers.sh

# The script will test multiple provider/model combinations in parallel
# Each test will:
# 1. Run a specific task for each provider/model
# 2. Save the output
# 3. Report success/failure
```

The tests ensure that:
- The provider can successfully connect to its API
- The model can process ARC-AGI tasks
- The output matches the expected format
- The provider correctly handles token usage and costs

### Testing Different Model Configurations

You can test the same model with different configurations by using the `--config` parameter:

```bash
# Test a model with a specific configuration
python3 -m main --data_dir data/arc-agi/data/evaluation --config claude_sonnet --task_id sample_task_id --print_logs
```

The `test_providers.sh` script includes examples of testing the same model with different configurations, such as:
- `openai o1 0b17323b high_temp` - Testing o1 with high temperature
- `openai o1 0b17323b low_temp` - Testing o1 with low temperature

## Adding New Providers and Models

### 1. Configure Models in models.yml

New models are defined in `src/arc_agi_benchmarking/models.yml`. Each model requires:

```yaml
models:
  - name: "model_config_name"     # A unique identifier for this model configuration
    model_name: "actual-model-name"  # The actual model name used by the provider's API
    provider: "provider-name"
    max_tokens: 4024  # or appropriate limit
    temperature: 0.0  # optional
    pricing:
      date: "YYYY-MM-DD"
      input: 0.00   # Cost per 1M input tokens
      output: 0.00  # Cost per 1M output tokens
```

### 2. Adding Configurations to Models

#### Multiple Configurations for the Same Model

In `models.yml`, you can create multiple configurations for the same underlying model by defining separate entries with different `name` values but the same `model_name`:

```yaml
models:
  # Configuration for short responses
  - name: "o1_short_response"
    model_name: "o1"
    provider: "openai"
    max_completion_tokens: 1024  # Shorter response limit
    pricing:
      date: "2025-02-23"
      input: 15.00
      output: 60.00

  # Configuration for long responses
  - name: "o1_long_response"
    model_name: "o1"
    provider: "openai"
    max_completion_tokens: 4024  # Longer response limit
    pricing:
      date: "2025-02-23"
      input: 15.00
      output: 60.00
```

When running the model, you specify the configuration name as the model parameter:

```bash
# Run with short response configuration
python3 -m main --config o1_short_response --task_id sample_task_id

# Run with long response configuration
python3 -m main --config o1_long_response --task_id sample_task_id
```

#### Using Model-Specific Parameters

You can add any model-specific parameters supported by the provider's API:

```yaml
models:
  - name: "gemini_pro"
    model_name: "gemini-1.5-pro"
    provider: "gemini"
    max_output_tokens: 4024  # Provider-specific parameter
    temperature: 0.0
    pricing:
      date: "2025-02-23"
      input: 1.25
      output: 5.00
```

Note how different providers may use different parameter names (e.g., `max_tokens`, `max_completion_tokens`, or `max_output_tokens`) depending on their API requirements.

#### Using Configurations in Batch Processing

When running batch tests with multiple configurations:

```bash
# Test with short response configuration
parallel --jobs 20 python3 -m main --data_dir data/arc-agi/data/evaluation --config o1_long_response --task_id {} --save_submission_dir submissions/o1_short :::: ./data/task_lists/public_evaluation_v1.txt

# Test with long response configuration
parallel --jobs 20 python3 -m main --data_dir data/arc-agi/data/evaluation --config o1_long_response --task_id {} --save_submission_dir submissions/o1_long :::: ./data/task_lists/public_evaluation_v1.txt
```

#### Comparing Configuration Results

After running tests with different configurations, you can compare their performance:

```bash
# Score short response configuration
python3 -m src/arc_agi_benchmarking.scoring.scoring --task_dir data/arc-agi/data/evaluation --submission_dir submissions/o1_short --print_logs --results_dir results/o1_short

# Score long response configuration
python3 -m src/arc_agi_benchmarking.scoring.scoring --task_dir data/arc-agi/data/evaluation --submission_dir submissions/o1_long --print_logs --results_dir results/o1_long
```

This allows you to systematically evaluate how different parameter settings affect model performance on ARC-AGI tasks.

### 3. Create Provider Adapter

1. Create a new file in `src/arc_agi_benchmarking/adapters/` (e.g., `my_provider.py`)
2. Implement the `ProviderAdapter` class:
   ```python
   from .provider import ProviderAdapter
   
   class MyProviderAdapter(ProviderAdapter):
       def init_client(self):
           # Initialize API client
           pass
           
       def make_prediction(self, prompt: str) -> Attempt:
           # Make prediction and return standardized Attempt object
           pass
           
       def chat_completion(self, messages: str) -> str:
           # Handle chat completion
           pass
   ```

3. Key requirements:
   - Handle authentication (typically via environment variables)
   - Implement proper error handling
   - Convert provider-specific responses to standardized formats
   - Track and report token usage and costs

### 4. Test New Provider

1. Add test cases to `test_providers.sh`
2. Test with sample tasks:
   ```bash
   python3 -m main --data_dir data/arc-agi/data/evaluation --provider new_provider --model new_model --task_id sample_task_id --print_logs
   ```

Remember to:
- Follow the existing patterns in other provider implementations
- Maintain consistent error handling
- Document any provider-specific requirements or limitations
- Update tests to cover the new provider

## Running Tests

To run the tests, execute the following command from the root directory:

```bash
pytest
```
