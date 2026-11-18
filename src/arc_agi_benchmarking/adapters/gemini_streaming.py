"""One Gemini prediction across streamed decode slices and checkpoint recovery."""

import base64
import copy
import logging
from contextlib import suppress
from dataclasses import dataclass, field
from types import SimpleNamespace

import httpx
from google.genai import types

logger = logging.getLogger(__name__)
MAX_STREAM_RECOVERY_RETRIES = 3


def _counts(metadata):
    prompt = metadata.get("promptTokenCount") or 0
    output = metadata.get("candidatesTokenCount") or 0
    thoughts = metadata.get("thoughtsTokenCount") or 0
    return {
        "prompt_token_count": prompt,
        "candidates_token_count": output,
        "thoughts_token_count": thoughts,
        "total_token_count": metadata.get(
            "totalTokenCount", prompt + output + thoughts
        ),
        "cached_content_token_count": metadata.get("cachedContentTokenCount") or 0,
    }


class GeminiStreamError(ValueError):
    """A failed prediction retaining reported usage and partial response data."""

    def __init__(self, message, slices, usage, cause=None):
        super().__init__(message)
        self.response = {"slices": slices, "reported_usage": dict(usage)}
        self.usage_metadata = SimpleNamespace(**usage)
        self.status_code = getattr(cause, "status_code", getattr(cause, "code", None))
        self.body = getattr(cause, "body", getattr(cause, "response_json", None))


@dataclass
class _Slice:
    parts: list = field(default_factory=list)
    pending: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    checkpoint_metadata: dict = field(default_factory=dict)
    token: bytes | None = None
    reason: str | None = None
    saw_chunk: bool = False

    def payload(self):
        candidate = {
            "content": {"role": "model", "parts": self.parts + self.pending},
            "finishReason": self.reason,
        }
        if self.token:
            candidate["continuationToken"] = base64.b64encode(self.token).decode()
        return {
            "candidates": [candidate],
            "usageMetadata": copy.deepcopy(self.metadata),
        }


def stream_with_continuation(adapter, contents, config_params):
    """Follow continuation checkpoints without creating a new benchmark attempt.

    Usage within a single HTTP stream is cumulative; usage across HTTP requests
    is additive. Uncommitted output rolls back after interruption, but observed
    billable usage never rolls back. Every request goes through the adapter's
    existing limiter and raw API logger.
    """
    params = copy.deepcopy(config_params)
    budget = params.get("max_output_tokens")
    explicit_budget = type(budget) is int and budget > 0
    # Keep all HTTP requests visible to the harness instead of letting the SDK
    # merge, reconnect, or retry them behind the limiter/raw logger.
    params["automatic_continuation"] = False
    if params.get("should_return_http_response"):
        raise ValueError("Raw HTTP response mode is not supported for streaming")
    config = types.GenerateContentConfig(**params)
    if config.continuation_token is not None:
        raise ValueError("Streaming continuation manages the continuation token")
    options = config.http_options or types.HttpOptions()
    if options.timeout is None:
        options.timeout = 3_600_000
    options.retry_options = types.HttpRetryOptions(attempts=1)
    extra = copy.deepcopy(options.extra_body or {})
    if any(
        key in extra for key in ("contents", "continuationToken", "generationConfig")
    ):
        raise ValueError(
            "Streaming continuation manages contents, token, and generationConfig"
        )
    options.extra_body = extra
    config.http_options = options
    base_contents = [
        c.model_dump(mode="json", by_alias=True, exclude_none=True) for c in contents
    ]
    parts, slices = [], []
    usage = _counts({})
    seen_tokens = set()
    recovery_retries = 0

    def failure(message, cause=None):
        return GeminiStreamError(message, slices, usage, cause)

    while True:
        hop = _Slice()
        stream = None
        stream_error = None
        try:
            stream = adapter._request(
                "models.generate_content_stream",
                adapter.client.models.generate_content_stream,
                model=adapter.model_config.model_name,
                contents=copy.deepcopy(contents),
                config=config.model_copy(deep=True),
                _raw_log_deferred=True,
            )
            for chunk in stream:
                hop.saw_chunk = True
                if chunk.usage_metadata is not None:
                    metadata = chunk.usage_metadata.model_dump(
                        mode="json",
                        by_alias=True,
                        exclude_none=True,
                    )
                    if metadata:
                        hop.metadata = metadata
                if not chunk.candidates:
                    continue
                candidate = chunk.candidates[0]
                if candidate.content and candidate.content.parts:
                    hop.pending.extend(
                        p.model_dump(mode="json", by_alias=True, exclude_none=True)
                        for p in candidate.content.parts
                    )
                # Tokens can arrive before the terminal finish reason. Only
                # output covered by the latest checkpoint is safe to replay.
                token = candidate.continuation_token
                if token and token != hop.token:
                    if token in seen_tokens or token == config.continuation_token:
                        raise ValueError(
                            "Gemini returned a repeated continuation token"
                        )
                    seen_tokens.add(token)
                    hop.token = token
                    hop.parts.extend(hop.pending)
                    hop.pending.clear()
                    hop.checkpoint_metadata = copy.deepcopy(hop.metadata)
                if candidate.finish_reason is not None:
                    hop.reason = candidate.finish_reason.value
        except BaseException as exc:
            # The per-request failure record includes the partial stream and its
            # observed usage, including on cancellation. Never log credentials.
            partial = GeminiStreamError(
                str(exc), [hop.payload()], _counts(hop.metadata), exc
            )
            adapter._record_deferred_raw_api_failure(stream, partial)
            if not isinstance(exc, Exception):
                raise
            stream_error = exc
        else:
            adapter._record_deferred_raw_api_success(stream, hop.payload())
        finally:
            if stream is not None and hasattr(stream, "close"):
                with suppress(Exception):
                    stream.close()

        # Charge the latest reported cumulative usage once for this request,
        # even when output must roll back to a checkpoint.
        for key, value in _counts(hop.metadata).items():
            usage[key] += value
        slices.append(hop.payload())
        if stream_error is not None and not isinstance(
            stream_error, httpx.TransportError
        ):
            raise failure(
                f"Gemini stream failed: {stream_error}", stream_error
            ) from stream_error

        cutoff = hop.reason is None and bool(hop.token or config.continuation_token)
        if stream_error is not None or cutoff:
            if hop.token and not hop.checkpoint_metadata:
                raise failure(
                    "Gemini stream checkpoint has no usage metadata"
                ) from stream_error
            if hop.saw_chunk and not hop.metadata:
                raise failure(
                    "Interrupted Gemini stream has no usage metadata"
                ) from stream_error
            if (hop.token or config.continuation_token) and not explicit_budget:
                raise failure(
                    "Gemini checkpoint recovery requires explicit positive max_output_tokens"
                )
            if hop.token:
                parts.extend(hop.parts)
                config.continuation_token = hop.token
            recovery_retries += 1
            if recovery_retries > MAX_STREAM_RECOVERY_RETRIES:
                raise failure(
                    "Gemini stream recovery retries exhausted"
                ) from stream_error
            logger.warning(
                "Gemini stream interrupted; resuming from %s checkpoint (%d/%d)",
                "latest" if config.continuation_token else "initial",
                recovery_retries,
                MAX_STREAM_RECOVERY_RETRIES,
            )
        else:
            logger.info(
                "Gemini stream slice %d finished with %s", len(slices), hop.reason
            )
            if not hop.metadata:
                raise failure("Gemini stream ended without usage metadata")
            if hop.reason not in ("CONTINUATION", "STOP", "MAX_TOKENS"):
                raise failure(f"Unexpected Gemini stream finish reason: {hop.reason}")
            if hop.reason == "CONTINUATION":
                if not explicit_budget:
                    raise failure(
                        "Gemini CONTINUATION requires explicit positive max_output_tokens"
                    )
                if not hop.token:
                    raise failure(
                        "Gemini CONTINUATION response has no continuation token"
                    )
                if not hop.checkpoint_metadata:
                    raise failure("Gemini stream checkpoint has no usage metadata")
                parts.extend(hop.parts)
                config.continuation_token = hop.token
            else:
                parts.extend(hop.parts + hop.pending)
                if hop.reason == "MAX_TOKENS":
                    logger.warning("Gemini exhausted its cumulative output budget")
                text = "".join(
                    p["text"]
                    for p in parts
                    if isinstance(p.get("text"), str) and not p.get("thought")
                )
                if not text.strip():
                    raise failure("Gemini stream completed without visible answer text")
                return SimpleNamespace(
                    text=text,
                    parts=parts,
                    slices=slices,
                    finish_reason=hop.reason,
                    usage_metadata=SimpleNamespace(**usage),
                )

        # Preserve the configured cumulative output budget and opaque native
        # parts/signatures. Do not reduce max_output_tokens between requests.
        options.extra_body = (
            {
                **extra,
                "contents": base_contents
                + [{"role": "model", "parts": copy.deepcopy(parts)}],
            }
            if config.continuation_token
            else extra
        )
