import os
from openai import OpenAI
from .openai_base import OpenAIBaseAdapter


class DeepseekAdapter(OpenAIBaseAdapter):
    """Adapter for Deepseek API."""

    def init_client(self):
        """Initialize the OpenAI client configured for Deepseek."""
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables")

        return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
