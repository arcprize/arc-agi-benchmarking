from openai import OpenAI
from .openai_base import OpenAIBaseAdapter


class MuleRouterAdapter(OpenAIBaseAdapter):
    """Adapter for MuleRouter API (Qwen models via OpenAI-compatible endpoint)."""

    def init_client(self):
        """Initialize the OpenAI client configured for MuleRouter API."""
        return OpenAI(
            api_key=self.get_api_key(),
            base_url="https://api.mulerouter.ai/vendors/openai/v1",
            max_retries=0,
        )
