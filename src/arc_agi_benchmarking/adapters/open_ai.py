import os
from openai import OpenAI
from .openai_base import OpenAIBaseAdapter


class OpenAIAdapter(OpenAIBaseAdapter):
    """Adapter for OpenAI API."""

    def init_client(self):
        """Initialize the OpenAI client.

        Honors optional `base_url` and `api_key_env` from the model config so the
        OpenAI adapter can target any OpenAI-compatible endpoint. Defaults preserve
        the standard OpenAI behavior (OPENAI_API_KEY + default base URL).
        """
        api_key_env = self.model_config.api_key_env or "OPENAI_API_KEY"
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"{api_key_env} not found in environment variables")

        client_kwargs = {"api_key": api_key, "max_retries": 0, "timeout": 1800}
        if self.model_config.base_url:
            client_kwargs["base_url"] = self.model_config.base_url

        return OpenAI(**client_kwargs)
