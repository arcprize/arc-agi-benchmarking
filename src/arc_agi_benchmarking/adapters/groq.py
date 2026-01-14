import os
from openai import OpenAI
from .openai_base import OpenAIBaseAdapter


class GroqAdapter(OpenAIBaseAdapter):
    """Adapter for Groq API."""

    def init_client(self):
        """Initialize the OpenAI client configured for Groq."""
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
