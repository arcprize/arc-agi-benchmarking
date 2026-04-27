import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from together import Together

from arc_agi_benchmarking.errors import TokenMismatchError
from arc_agi_benchmarking.schemas import (
    Attempt,
    AttemptMetadata,
    Choice,
    CompletionTokensDetails,
    Cost,
    Message,
    Usage,
)
from arc_agi_benchmarking.utils.parsing import parse_and_validate_json

from .provider import ProviderAdapter

_CONFIG_ONLY_KWARGS = {"rate_limit", "pricing", "enable_thinking"}


def _filter_api_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in kwargs.items() if k not in _CONFIG_ONLY_KWARGS}


def _get_value(obj: Any, attr: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


class TogetherAdapter(ProviderAdapter):
    """Adapter boundary for the official Together SDK."""

    def init_client(self):
        api_key = os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY not found in environment variables")

        return Together(api_key=api_key)

    def _call_together_model(self, prompt: str) -> Any:
        api_kwargs = _filter_api_kwargs(self.model_config.kwargs)
        if api_kwargs.get("stream"):
            raise ValueError("TogetherAdapter streaming is not supported yet")

        messages = [{"role": "user", "content": prompt}]
        return self.client.chat.completions.create(
            model=self.model_config.model_name,
            messages=messages,
            **api_kwargs,
        )

    def make_prediction(
        self,
        prompt: str,
        task_id: Optional[str] = None,
        test_id: Optional[str] = None,
        pair_index: int = None,
    ) -> Attempt:
        start_time = datetime.now(timezone.utc)
        response = self._call_together_model(prompt)
        end_time = datetime.now(timezone.utc)

        content = self._get_content(response)
        usage = self._get_usage(response)
        cost = self._calculate_cost(usage)

        choices = [
            Choice(index=0, message=Message(role="user", content=prompt)),
            Choice(
                index=1,
                message=Message(role=self._get_role(response), content=content),
            ),
        ]

        metadata = AttemptMetadata(
            model=self.model_config.model_name,
            provider=self.model_config.provider,
            start_timestamp=start_time,
            end_timestamp=end_time,
            choices=choices,
            reasoning_summary=self._get_reasoning_summary(response),
            kwargs=self.model_config.kwargs,
            usage=usage,
            cost=cost,
            task_id=task_id,
            pair_index=pair_index,
            test_id=test_id,
        )

        return Attempt(metadata=metadata, answer=content)

    def extract_json_from_response(
        self, input_response: str
    ) -> Optional[list[list[int]]]:
        try:
            return parse_and_validate_json(input_response)
        except ValueError:
            return None

    def _first_choice(self, response: Any) -> Any:
        choices = _get_value(response, "choices", []) or []
        return choices[0] if choices else None

    def _get_content(self, response: Any) -> str:
        choice = self._first_choice(response)
        message = _get_value(choice, "message")
        content = _get_value(message, "content")
        if content is None:
            content = _get_value(choice, "text", "")
        return (content or "").strip()

    def _get_role(self, response: Any) -> str:
        choice = self._first_choice(response)
        message = _get_value(choice, "message")
        return _get_value(message, "role", "assistant") or "assistant"

    def _get_reasoning_summary(self, response: Any) -> Optional[str]:
        choice = self._first_choice(response)
        message = _get_value(choice, "message")
        return _get_value(message, "reasoning")

    def _get_usage(self, response: Any) -> Usage:
        raw_usage = _get_value(response, "usage")
        if not raw_usage:
            return self._zero_usage()

        prompt_tokens = int(_get_value(raw_usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(_get_value(raw_usage, "completion_tokens", 0) or 0)
        total_tokens = int(
            _get_value(raw_usage, "total_tokens", prompt_tokens + completion_tokens)
            or 0
        )
        explicit_reasoning_tokens = self._get_explicit_reasoning_tokens(raw_usage)

        if total_tokens < prompt_tokens + completion_tokens:
            raise TokenMismatchError(
                f"Token count mismatch: Together reports total {total_tokens}, "
                f"but prompt_tokens {prompt_tokens} + completion_tokens {completion_tokens} "
                f"= {prompt_tokens + completion_tokens}"
            )

        if total_tokens > prompt_tokens + completion_tokens:
            inferred_reasoning_tokens = total_tokens - (
                prompt_tokens + completion_tokens
            )
            if (
                explicit_reasoning_tokens
                and explicit_reasoning_tokens != inferred_reasoning_tokens
            ):
                raise TokenMismatchError(
                    f"Token count mismatch: Together reports total {total_tokens}, "
                    f"but prompt_tokens {prompt_tokens} + completion_tokens "
                    f"{completion_tokens} + reasoning_tokens "
                    f"{explicit_reasoning_tokens} = "
                    f"{prompt_tokens + completion_tokens + explicit_reasoning_tokens}"
                )
            reasoning_tokens = inferred_reasoning_tokens
        else:
            reasoning_tokens = explicit_reasoning_tokens
            if reasoning_tokens > completion_tokens:
                raise TokenMismatchError(
                    f"Token count mismatch: Together reports completion_tokens "
                    f"{completion_tokens}, but reasoning_tokens {reasoning_tokens} "
                    "cannot exceed completion_tokens when total_tokens equals "
                    "prompt_tokens + completion_tokens"
                )

        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            completion_tokens_details=CompletionTokensDetails(
                reasoning_tokens=reasoning_tokens,
                accepted_prediction_tokens=completion_tokens,
                rejected_prediction_tokens=0,
            ),
        )

    def _get_explicit_reasoning_tokens(self, raw_usage: Any) -> int:
        direct_reasoning_tokens = _get_value(raw_usage, "reasoning_tokens", 0) or 0
        if direct_reasoning_tokens:
            return int(direct_reasoning_tokens)

        completion_details = _get_value(raw_usage, "completion_tokens_details")
        completion_reasoning_tokens = (
            _get_value(completion_details, "reasoning_tokens", 0) or 0
        )
        if completion_reasoning_tokens:
            return int(completion_reasoning_tokens)

        output_details = _get_value(raw_usage, "output_tokens_details")
        output_reasoning_tokens = _get_value(output_details, "reasoning_tokens", 0) or 0
        return int(output_reasoning_tokens)

    def _calculate_cost(self, usage: Usage) -> Cost:
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens
        reasoning_tokens = usage.completion_tokens_details.reasoning_tokens

        prompt_cost = prompt_tokens * (self.model_config.pricing.input / 1_000_000)

        if total_tokens == prompt_tokens + completion_tokens:
            completion_tokens_for_cost = max(0, completion_tokens - reasoning_tokens)
        else:
            completion_tokens_for_cost = completion_tokens

        completion_cost = completion_tokens_for_cost * (
            self.model_config.pricing.output / 1_000_000
        )
        reasoning_cost = reasoning_tokens * (
            self.model_config.pricing.output / 1_000_000
        )
        total_cost = prompt_cost + completion_cost + reasoning_cost

        return Cost(
            prompt_cost=prompt_cost,
            completion_cost=completion_cost,
            reasoning_cost=reasoning_cost,
            total_cost=total_cost,
        )

    def _zero_usage(self) -> Usage:
        return Usage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            completion_tokens_details=CompletionTokensDetails(
                reasoning_tokens=0,
                accepted_prediction_tokens=0,
                rejected_prediction_tokens=0,
            ),
        )
