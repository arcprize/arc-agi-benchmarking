from openai import OpenAI
from .openai_base import OpenAIBaseAdapter


class DeepseekAdapter(OpenAIBaseAdapter):
    """Adapter for Deepseek API."""

    def init_client(self):
        """Initialize the OpenAI client configured for Deepseek."""
        return OpenAI(api_key=self.get_api_key(), base_url="https://api.deepseek.com", max_retries=0)
