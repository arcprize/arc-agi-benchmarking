import os
from typing import List, Dict, Any
from openai import OpenAI
from .openai_base import OpenAIBaseAdapter, _filter_api_kwargs
import logging

logger = logging.getLogger(__name__)


class FireworksAdapter(OpenAIBaseAdapter):
    """Adapter for Fireworks API."""

    def init_client(self):
        """Initialize the OpenAI client configured for Fireworks."""
        api_key = os.environ.get("FIREWORKS_API_KEY")
        if not api_key:
            raise ValueError("FIREWORKS_API_KEY not found in environment variables")

        return OpenAI(api_key=api_key, base_url="https://api.fireworks.ai/inference/v1")

    def _prepare_kwargs(self) -> dict:
        api_kwargs = _filter_api_kwargs(self.model_config.kwargs)
        api_kwargs["store"] = False
        return api_kwargs

    def _chat_completion(self, messages: List[Dict[str, str]]) -> Any:
        api_kwargs = self._prepare_kwargs()

        logger.debug(
            f"Calling Fireworks API with model: {self.model_config.model_name} and kwargs: {api_kwargs}"
        )
        return self.client.chat.completions.create(
            model=self.model_config.model_name,
            messages=messages,
            **api_kwargs,
        )

    def _chat_completion_stream(self, messages: List[Dict[str, str]]) -> Any:
        api_kwargs = self._prepare_kwargs()
        stream_kwargs = {k: v for k, v in api_kwargs.items() if k != "stream"}

        logger.debug(
            f"Starting streaming Fireworks API call with model: {self.model_config.model_name}"
        )
        try:
            stream = self.client.chat.completions.create(
                model=self.model_config.model_name,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                **stream_kwargs,
            )

            from openai.types.chat import ChatCompletion, ChatCompletionMessage
            from openai.types.chat.chat_completion import Choice as OpenAIChoice
            from openai.types import CompletionUsage
            import time

            content_chunks = []
            last_chunk = None
            finish_reason = "stop"
            chunk_count = 0

            for chunk in stream:
                last_chunk = chunk
                chunk_count += 1

                if chunk.choices:
                    delta_content = chunk.choices[0].delta.content or ""
                    if delta_content:
                        content_chunks.append(delta_content)

                if chunk.choices and chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

            final_content = "".join(content_chunks)
            usage_data = (
                last_chunk.usage if last_chunk and hasattr(last_chunk, "usage") else None
            )
            response_id = last_chunk.id if last_chunk else f"stream-{int(time.time())}"

            if not usage_data:
                logger.warning("No usage data received from streaming response")
                usage_data = CompletionUsage(
                    prompt_tokens=0, completion_tokens=0, total_tokens=0
                )

            return ChatCompletion(
                id=response_id,
                choices=[
                    OpenAIChoice(
                        finish_reason=finish_reason,
                        index=0,
                        message=ChatCompletionMessage(
                            content=final_content, role="assistant"
                        ),
                        logprobs=None,
                    )
                ],
                created=int(time.time()),
                model=self.model_config.model_name,
                object="chat.completion",
                usage=usage_data,
            )

        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            raise
