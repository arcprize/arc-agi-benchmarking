import os
from openai import OpenAI
from .openai_base import OpenAIBaseAdapter


class GrokAdapter(OpenAIBaseAdapter):
    """Adapter for Grok API (via x.ai)."""

    def init_client(self):
        """Initialize the OpenAI client configured for Grok API."""
        api_key = os.environ.get("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY not found in environment variables")

        return OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
