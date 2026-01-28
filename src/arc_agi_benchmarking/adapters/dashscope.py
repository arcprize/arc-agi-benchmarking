import os
from openai import OpenAI
from .openai_base import OpenAIBaseAdapter


class DashScopeAdapter(OpenAIBaseAdapter):
    """Adapter for Alibaba Cloud DashScope API (Qwen models).

    WARNING: Data retention policy for DashScope is unclear. Only use this
    adapter with PUBLIC data. Do not send private or sensitive data through
    this provider until you have verified their data handling policies.
    """

    def init_client(self):
        """Initialize the OpenAI client configured for DashScope API."""
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY not found in environment variables")

        # Use international endpoint by default
        # China: https://dashscope.aliyuncs.com/compatible-mode/v1
        # Singapore: https://dashscope-intl.aliyuncs.com/compatible-mode/v1
        base_url = os.environ.get(
            "DASHSCOPE_BASE_URL",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )

        return OpenAI(api_key=api_key, base_url=base_url)
