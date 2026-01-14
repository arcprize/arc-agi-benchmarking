import os
from openai import OpenAI
from .openai_base import OpenAIBaseAdapter


class OpenAIAdapter(OpenAIBaseAdapter):
    """Adapter for OpenAI API."""

    def init_client(self):
        """Initialize the OpenAI client."""
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        return OpenAI(
            max_retries=0,
            timeout=1800,
        )
