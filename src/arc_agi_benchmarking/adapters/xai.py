import os
import httpx
from openai import OpenAI
from .openai_base import OpenAIBaseAdapter


class XAIAdapter(OpenAIBaseAdapter):
    """Adapter for XAI API."""

    def init_client(self):
        """Initialize the OpenAI client configured for XAI API."""
        api_key = os.environ.get("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY not found in environment variables")

        return OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
            timeout=httpx.Timeout(3600, connect=30)
        )
