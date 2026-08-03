import httpx
from openai import OpenAI
from .openai_base import OpenAIBaseAdapter


class XAIAdapter(OpenAIBaseAdapter):
    """Adapter for XAI API."""

    def init_client(self):
        """Initialize the OpenAI client configured for XAI API."""
        return OpenAI(
            api_key=self.get_api_key(),
            base_url="https://api.x.ai/v1",
            timeout=httpx.Timeout(3600, connect=30),
            max_retries=0,
        )
