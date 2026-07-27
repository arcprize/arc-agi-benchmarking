from openai import OpenAI
from .openai_base import OpenAIBaseAdapter


class GrokAdapter(OpenAIBaseAdapter):
    """Adapter for Grok API (via x.ai)."""

    def init_client(self):
        """Initialize the OpenAI client configured for Grok API."""
        return OpenAI(api_key=self.get_api_key(), base_url="https://api.x.ai/v1")
