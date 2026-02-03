import os
from openai import OpenAI
from .openai_base import OpenAIBaseAdapter


class MuleRouterAdapter(OpenAIBaseAdapter):
    """Adapter for MuleRouter API (Qwen models via OpenAI-compatible endpoint)."""

    def init_client(self):
        """Initialize the OpenAI client configured for MuleRouter API."""
        api_key = os.environ.get("MULEROUTER_API_KEY")
        if not api_key:
            raise ValueError("MULEROUTER_API_KEY not found in environment variables")

        return OpenAI(
            api_key=api_key, base_url="https://api.mulerouter.ai/vendors/openai/v1"
        )
