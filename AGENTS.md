# AGENTS.md

This file provides context for AI coding agents (Codex, etc.) when working on this repository.

## Project Overview

ARC-AGI benchmarking framework for evaluating LLMs on ARC (Abstraction and Reasoning Corpus) pattern recognition tasks. Supports multiple providers (OpenAI, Anthropic, Gemini, Fireworks, Grok, etc.) with built-in rate limiting, retries, and scoring.

## Quick Commands

```bash
make install      # Install package in editable mode
make test         # Run all tests
make test-verbose # Run tests with verbose output
make run-sample   # Run random baseline on single sample task
make run-batch    # Run random baseline on all sample tasks
make score        # Score submissions against ground truth
make clean        # Remove generated files and caches
```

Or manually:

```bash
# Install
pip install -e .

# Run tests
pytest -q                    # Quick
pytest -v                    # Verbose
pytest -x                    # Stop on first failure

# Run sample benchmark (no API key needed)
python main.py --data_dir data/sample/tasks --config random-baseline --task_id 66e6c45b --save_submission_dir submissions/test

# Run batch benchmark
python cli/run_all.py --data_dir data/sample/tasks --config random-baseline --save_submission_dir submissions/test

# Score submissions
python src/arc_agi_benchmarking/scoring/scoring.py --task_dir data/sample/tasks --submission_dir submissions/test

# View task visually
python -m arc_agi_benchmarking.utils --task data/sample/tasks/66e6c45b.json
```

## Project Structure

```
├── main.py                     # Single-task runner
├── cli/
│   ├── run_all.py              # Batch runner (main entry point for benchmarks)
│   └── submission_cli.py       # Submission management CLI
├── src/arc_agi_benchmarking/
│   ├── adapters/               # Provider adapters (one per API)
│   │   ├── provider.py         # Base ProviderAdapter interface
│   │   ├── openai_base.py      # Shared OpenAI-compatible base class
│   │   ├── open_ai.py          # OpenAI adapter
│   │   ├── anthropic.py        # Anthropic adapter
│   │   ├── gemini.py           # Google Gemini adapter
│   │   └── ...                 # Other providers
│   ├── scoring/
│   │   └── scoring.py          # Submission scoring logic
│   ├── utils/
│   │   ├── preflight.py        # Pre-run validation & cost estimation
│   │   ├── viewer.py           # Terminal task visualization
│   │   ├── rate_limiter.py     # API rate limiting
│   │   └── metrics.py          # Metrics collection
│   ├── tests/                  # All tests live here
│   └── models.yml              # Model configurations
├── data/
│   ├── sample/tasks/           # Sample ARC tasks for testing
│   └── v2/                     # Full evaluation set (clone separately)
└── provider_config.yml         # Rate limits per provider
```

## Key Files

- **models.yml** (`src/arc_agi_benchmarking/models.yml`): Model configurations including pricing, tokens, temperature
- **provider_config.yml**: Rate limits (requests/period) per provider
- **.env**: API keys (copy from `.env.example`)

## Architecture

### Adapter Pattern
All providers implement `ProviderAdapter` interface from `adapters/provider.py`:
- `get_response(system_prompt, user_prompt) -> str`: Main API call
- OpenAI-compatible providers extend `OpenAIBaseAdapter` for shared logic

### Adding a New Provider
1. Create `adapters/<provider>.py` extending `ProviderAdapter` or `OpenAIBaseAdapter`
2. Export in `adapters/__init__.py`
3. Add branch in `main.py` to recognize the provider
4. Add model config in `models.yml`

### Task Format
ARC tasks are JSON with:
- `train`: List of input/output grid pairs (examples)
- `test`: List of test cases with input grids (and expected outputs for scoring)

Grids are 2D arrays of integers 0-9 representing colors.

## Testing

Tests are in `src/arc_agi_benchmarking/tests/`. Run with pytest:
```bash
pytest                          # All tests
pytest tests/test_preflight.py  # Specific file
pytest -k "test_cost"           # By name pattern
```

## Environment Variables

Required API keys (set in `.env`):
- `OPENAI_API_KEY` - OpenAI
- `ANTHROPIC_API_KEY` - Anthropic
- `GOOGLE_API_KEY` - Google Gemini
- `XAI_API_KEY` - X.AI / Grok
- `FIREWORKS_API_KEY` - Fireworks
- `GROQ_API_KEY` - Groq
- `OPENROUTER_API_KEY` - OpenRouter
- `HUGGING_FACE_API_KEY` - HuggingFace

## Code Conventions

- Use `python -m pytest` for running tests
- Adapters should handle their own error cases and return clean strings
- Preflight validation runs automatically before benchmarks (skip with `--skip-preflight`)
- Submissions are saved as JSON with `attempt_1`, `attempt_2` keys per task
