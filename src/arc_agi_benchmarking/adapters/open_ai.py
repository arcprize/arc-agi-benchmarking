from openai import OpenAI
from .openai_base import OpenAIBaseAdapter


OPENAI_REQUEST_TIMEOUT_SECONDS = 2 * 60 * 60


class OpenAIAdapter(OpenAIBaseAdapter):
    """Adapter for OpenAI API."""

    def init_client(self):
        """Initialize the OpenAI client.

        Honors `base_url` and `api_key_env` from the model config so the OpenAI
        adapter can target OpenAI or any OpenAI-compatible endpoint without
        assuming which credential should be used.
        """
        client_kwargs = {
            "api_key": self.get_api_key(),
            "max_retries": 0,
            "timeout": OPENAI_REQUEST_TIMEOUT_SECONDS,
        }
        if self.model_config.base_url:
            client_kwargs["base_url"] = self.model_config.base_url

        return OpenAI(**client_kwargs)
