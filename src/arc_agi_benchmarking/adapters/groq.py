from openai import OpenAI
from .openai_base import OpenAIBaseAdapter


class GroqAdapter(OpenAIBaseAdapter):
    """Adapter for Groq API."""

    def init_client(self):
        """Initialize the OpenAI client configured for Groq."""
        return OpenAI(api_key=self.get_api_key(), base_url="https://api.groq.com/openai/v1", max_retries=0)
