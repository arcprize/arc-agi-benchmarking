import os
from openai import OpenAI
from .openai_base import OpenAIBaseAdapter


class OpenRouterAdapter(OpenAIBaseAdapter):
    """Adapter for OpenRouter API."""

    def init_client(self):
        """Initialize the OpenAI client configured for OpenRouter API."""
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")

        return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
