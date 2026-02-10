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

    def _chat_completion(self, messages: List[Dict[str, str]]) -> Any:
        api_kwargs = _filter_api_kwargs(self.model_config.kwargs)
        api_kwargs["store"] = False
        logger.debug(
            f"Calling Fireworks API with model: {self.model_config.model_name} and kwargs: {api_kwargs}"
        )
        return self.client.chat.completions.create(
            model=self.model_config.model_name, messages=messages, **api_kwargs
        )
