from openai import OpenAI
from .openai_base import OpenAIBaseAdapter


class OpenRouterAdapter(OpenAIBaseAdapter):
    """Adapter for OpenRouter API."""

    def init_client(self):
        """Initialize the OpenAI client configured for OpenRouter API."""
        return OpenAI(api_key=self.get_api_key(), base_url="https://openrouter.ai/api/v1")
