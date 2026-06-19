"""Anthropic Batch API runner for ARC benchmarking.

Submits all task predictions as a single Anthropic message batch,
polls for completion, retrieves results, and saves submissions.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import anthropic

from arc_agi_benchmarking.checkpoint import BatchProgressManager
from arc_agi_benchmarking.prompts.prompt_manager import convert_task_pairs_to_prompt
from arc_agi_benchmarking.schemas import (
    ARCPair,
    Attempt,
    AttemptMetadata,
    Choice,
    CompletionTokensDetails,
    Cost,
    Message,
    Usage,
)
import arc_agi_benchmarking.utils as utils

logger = logging.getLogger(__name__)

CUSTOM_ID_SEP = "__"
BATCH_STATE_FILE = "anthropic_batch_state.json"
POLL_INTERVAL_SECONDS = 60


def _encode_custom_id(task_id: str, pair_index: int, attempt_num: int) -> str:
    return f"{task_id}{CUSTOM_ID_SEP}{pair_index}{CUSTOM_ID_SEP}{attempt_num}"


def _decode_custom_id(custom_id: str) -> tuple:
    parts = custom_id.split(CUSTOM_ID_SEP)
    return parts[0], int(parts[1]), int(parts[2])


def _build_request_params(model_config, prompt: str) -> dict:
    """Build the params dict for a single batch request from model config."""
    params = {
        "model": model_config.model_name,
        "max_tokens": model_config.kwargs.get("max_tokens", 128000),
        "messages": [{"role": "user", "content": prompt}],
    }

    # Add thinking config
    thinking = model_config.kwargs.get("thinking")
    if thinking:
        params["thinking"] = thinking

    # Add extra_body fields (e.g. output_config) directly into params
    extra_body = model_config.kwargs.get("extra_body")
    if extra_body:
        params.update(extra_body)

    return params


def _response_to_attempt(
    result_msg,
    model_config,
    task_id: str,
    test_id: str,
    pair_index: int,
) -> Attempt:
    """Convert an Anthropic batch result message to an Attempt object."""
    input_cost_per_token = model_config.pricing.input / 1_000_000
    output_cost_per_token = model_config.pricing.output / 1_000_000

    prompt_cost = result_msg.usage.input_tokens * input_cost_per_token
    completion_cost = result_msg.usage.output_tokens * output_cost_per_token

    # Extract text answer (skip thinking blocks)
    answer = ""
    for content in result_msg.content:
        if content.type == "text":
            answer = content.text
            break

    # Extract thinking summary
    thinking_texts = []
    for block in result_msg.content:
        if hasattr(block, "type") and block.type == "thinking" and hasattr(block, "thinking"):
            if isinstance(block.thinking, str):
                thinking_texts.append(block.thinking)
    reasoning_summary = "\n\n".join(thinking_texts) if thinking_texts else None

    metadata = AttemptMetadata(
        model=model_config.model_name,
        provider=model_config.provider,
        start_timestamp=datetime.now(timezone.utc),
        end_timestamp=datetime.now(timezone.utc),
        choices=[
            Choice(index=0, message=Message(role="user", content="[batch request]")),
            Choice(index=1, message=Message(role="assistant", content=answer)),
        ],
        kwargs=model_config.kwargs,
        reasoning_summary=reasoning_summary,
        usage=Usage(
            prompt_tokens=result_msg.usage.input_tokens,
            completion_tokens=result_msg.usage.output_tokens,
            total_tokens=result_msg.usage.input_tokens + result_msg.usage.output_tokens,
            completion_tokens_details=CompletionTokensDetails(
                reasoning_tokens=0,
                accepted_prediction_tokens=result_msg.usage.output_tokens,
                rejected_prediction_tokens=0,
            ),
        ),
        cost=Cost(
            prompt_cost=prompt_cost,
            completion_cost=completion_cost,
            total_cost=prompt_cost + completion_cost,
        ),
        task_id=task_id,
        pair_index=pair_index,
        test_id=test_id,
    )

    return Attempt(metadata=metadata, answer=answer)


def _load_batch_state(checkpoint_dir: Path) -> Optional[dict]:
    state_file = checkpoint_dir / BATCH_STATE_FILE
    if state_file.exists():
        return json.loads(state_file.read_text())
    return None


def _save_batch_state(checkpoint_dir: Path, state: dict):
    state_file = checkpoint_dir / BATCH_STATE_FILE
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, indent=2))


def _clear_batch_state(checkpoint_dir: Path):
    state_file = checkpoint_dir / BATCH_STATE_FILE
    if state_file.exists():
        state_file.unlink()


async def run_anthropic_batch(
    config_name: str,
    task_ids: List[str],
    data_dir: str,
    save_submission_dir: str,
    num_attempts: int,
    retry_attempts: int,
    progress_manager: BatchProgressManager,
    overwrite_submission: bool = False,
) -> int:
    """Run all tasks via Anthropic Batch API.

    Returns exit code (0 = success, 1 = failure).
    """
    model_config = utils.read_models_config(config_name)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
    client = anthropic.Anthropic(api_key=api_key)

    betas = model_config.kwargs.get("betas", [])
    checkpoint_dir = Path(save_submission_dir) / ".checkpoints"

    # Check for an in-progress batch from a previous run
    batch_state = _load_batch_state(checkpoint_dir)
    if batch_state and batch_state.get("status") != "ended":
        batch_id = batch_state["batch_id"]
        logger.info(f"Resuming existing batch: {batch_id}")
    else:
        # Build all requests
        requests = []
        for task_id in task_ids:
            # Skip if submission already exists
            if not overwrite_submission and utils.submission_exists(save_submission_dir, task_id):
                logger.info(f"Submission for {task_id} already exists, skipping")
                continue

            train_pairs = utils.get_train_pairs_from_task(data_dir, task_id)
            test_input_pairs = utils.get_test_input_from_task(data_dir, task_id)

            for pair_index, pair_input_obj in enumerate(test_input_pairs):
                prompt = convert_task_pairs_to_prompt(train_pairs, pair_input_obj)
                params = _build_request_params(model_config, prompt)

                for attempt_num in range(1, num_attempts + 1):
                    custom_id = _encode_custom_id(task_id, pair_index, attempt_num)
                    requests.append({
                        "custom_id": custom_id,
                        "params": params,
                    })

        if not requests:
            logger.info("No requests to submit. All tasks already have submissions.")
            return 0

        logger.info(f"Submitting batch with {len(requests)} requests...")

        # Submit batch (chunk if > 10000)
        if betas:
            batch = client.beta.messages.batches.create(
                requests=requests,
                betas=betas,
            )
        else:
            batch = client.messages.batches.create(requests=requests)

        batch_id = batch.id
        logger.info(f"Batch submitted: {batch_id}")

        _save_batch_state(checkpoint_dir, {
            "batch_id": batch_id,
            "status": "in_progress",
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "num_requests": len(requests),
        })

    # Poll for completion
    logger.info(f"Polling batch {batch_id} for completion...")
    while True:
        if betas:
            batch_status = client.beta.messages.batches.retrieve(batch_id, betas=betas)
        else:
            batch_status = client.messages.batches.retrieve(batch_id)

        status = batch_status.processing_status
        rc = batch_status.request_counts
        total = rc.processing + rc.succeeded + rc.errored + rc.canceled + rc.expired
        done = rc.succeeded + rc.errored + rc.canceled + rc.expired
        pct = done * 100 // total if total else 0
        logger.info(
            f"Batch {batch_id} | {status} | "
            f"Progress: {done}/{total} ({pct}%) | "
            f"Succeeded: {rc.succeeded} | Errored: {rc.errored} | "
            f"Processing: {rc.processing}"
        )

        if status == "ended":
            break

        _save_batch_state(checkpoint_dir, {
            "batch_id": batch_id,
            "status": status,
            "submitted_at": _load_batch_state(checkpoint_dir).get("submitted_at"),
        })

        await asyncio.sleep(POLL_INTERVAL_SECONDS)

    # Retrieve results
    logger.info(f"Batch {batch_id} complete. Retrieving results...")
    if betas:
        results_iter = client.beta.messages.batches.results(batch_id, betas=betas)
    else:
        results_iter = client.messages.batches.results(batch_id)

    # Group results by task_id
    task_results: Dict[str, Dict] = {}  # task_id -> {pair_index -> {attempt_key -> attempt_obj}}

    for entry in results_iter:
        custom_id = entry.custom_id
        task_id, pair_index, attempt_num = _decode_custom_id(custom_id)

        if task_id not in task_results:
            task_results[task_id] = {}
        if pair_index not in task_results[task_id]:
            task_results[task_id][pair_index] = {}

        attempt_key = f"attempt_{attempt_num}"

        if entry.result.type == "succeeded":
            result_msg = entry.result.message

            # Extract text answer (skip thinking blocks)
            answer_text = ""
            for content in result_msg.content:
                if content.type == "text":
                    answer_text = content.text
                    break

            if not answer_text.strip():
                logger.warning(f"Empty response for {custom_id}")
                task_results[task_id][pair_index][attempt_key] = None
                continue

            try:
                attempt_obj = _response_to_attempt(
                    result_msg, model_config, task_id, config_name, pair_index
                )

                # Parse JSON answer
                try:
                    if isinstance(attempt_obj.answer, str):
                        parsed = json.loads(attempt_obj.answer)
                        attempt_obj.answer = parsed
                except (json.JSONDecodeError, ValueError):
                    # Try provider-level extraction
                    from arc_agi_benchmarking.adapters.anthropic import AnthropicAdapter
                    adapter = AnthropicAdapter(config_name)
                    try:
                        parsed = adapter.extract_json_from_response(attempt_obj.answer)
                        if parsed is not None:
                            attempt_obj.answer = parsed
                        else:
                            logger.warning(f"Failed to parse response for {custom_id}")
                            attempt_obj.answer = None
                    except Exception as parse_err:
                        logger.warning(f"Parse extraction failed for {custom_id}: {parse_err}")
                        attempt_obj.answer = None

                # Check correctness
                test_pairs = utils.get_test_pairs_from_task(data_dir, task_id)
                if attempt_obj.answer is not None and pair_index < len(test_pairs):
                    attempt_obj.correct = attempt_obj.answer == test_pairs[pair_index].output

                task_results[task_id][pair_index][attempt_key] = attempt_obj.model_dump(mode="json")

            except Exception as e:
                logger.error(f"Error processing result for {custom_id}: {e}")
                task_results[task_id][pair_index][attempt_key] = None
        else:
            error_info = entry.result
            logger.warning(f"Request {custom_id} failed: {error_info.type} - {getattr(error_info, 'error', 'unknown')}")
            task_results[task_id][pair_index][attempt_key] = None

    # Save submissions
    successful = 0
    failed = 0
    for task_id, pairs in task_results.items():
        task_submission = []
        has_valid_attempt = False

        for pair_index in sorted(pairs.keys()):
            pair_data = pairs[pair_index]
            if any(v is not None for v in pair_data.values()):
                has_valid_attempt = True
            task_submission.append(pair_data)

        if has_valid_attempt:
            submission_path = Path(save_submission_dir) / f"{task_id}.json"
            submission_path.parent.mkdir(parents=True, exist_ok=True)
            with open(submission_path, "w") as f:
                json.dump(task_submission, f, indent=2)
            progress_manager.mark_completed(task_id)
            successful += 1
        else:
            progress_manager.mark_failed(task_id, "All attempts failed in batch")
            failed += 1

    logger.info(f"Batch processing complete: {successful} tasks saved, {failed} tasks failed")

    _clear_batch_state(checkpoint_dir)

    return 0 if failed == 0 else 1
