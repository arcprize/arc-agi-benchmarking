import os
from openai import OpenAI
from .openai_base import OpenAIBaseAdapter


class FireworksAdapter(OpenAIBaseAdapter):
    """Adapter for Fireworks API."""

    def init_client(self):
        """Initialize the OpenAI client configured for Fireworks."""
        api_key = os.environ.get("FIREWORKS_API_KEY")
        if not api_key:
            raise ValueError("FIREWORKS_API_KEY not found in environment variables")

        return OpenAI(api_key=api_key, base_url="https://api.fireworks.ai/inference/v1")
