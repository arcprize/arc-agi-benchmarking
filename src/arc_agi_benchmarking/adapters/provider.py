import abc
import os
from typing import List, Dict, Tuple, Any, Optional
import json
import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Callable, TypeVar
from arc_agi_benchmarking.schemas import Attempt, ModelConfig
from arc_agi_benchmarking.utils.task_utils import read_models_config
from arc_agi_benchmarking.utils.rate_limiter import RequestRateLimiter
from arc_agi_benchmarking.utils.logging_utils import (
    RawAPILogger,
    RawAPIRequestHandle,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")

class ProviderAdapter(abc.ABC):
    request_limit_scope = "api_request"

    def __init__(
        self,
        config: str,
        request_limiter: Optional[RequestRateLimiter] = None,
        raw_api_logger: Optional[RawAPILogger] = None,
    ):
        """
        Initialize the provider adapter with model configuration.
        
        Args:
            config: Configuration name that identifies the model and its settings
        """
        self.config = config
        self.model_config: ModelConfig = read_models_config(config)
        self.request_limiter = request_limiter
        self.raw_api_logger = raw_api_logger
        self._raw_api_context: Dict[str, Any] = {}
        self._deferred_raw_api_requests: Dict[int, RawAPIRequestHandle] = {}
        
        # Verify the provider matches the adapter
        adapter_provider = self.__class__.__name__.lower().replace('adapter', '')
        if adapter_provider != self.model_config.provider:
            raise ValueError(f"Model provider mismatch. Config '{config}' is for provider '{self.model_config.provider}' but was passed to {self.__class__.__name__}")
        
        # Initialize the client
        self.client = self.init_client()

        if request_limiter is not None and self.request_limit_scope != "api_request":
            logger.warning(
                "%s hides its internal HTTP traffic; rate limiting applies to each "
                "adapter invocation, not every underlying API request.",
                self.__class__.__name__,
            )

    @contextmanager
    def request_context(self, **context: Any):
        """Temporarily attach benchmark context to outbound API operations."""
        previous = getattr(self, "_raw_api_context", {})
        self._raw_api_context = {**previous, **context}
        try:
            yield
        finally:
            self._raw_api_context = previous

    def _request(
        self,
        operation: str,
        call: Callable[..., T],
        *args,
        _raw_log_deferred: bool = False,
        _raw_log_request: Any = None,
        **kwargs,
    ) -> T:
        """Acquire allowance immediately before invoking an outbound SDK call."""
        request_limiter = getattr(self, "request_limiter", None)
        if request_limiter is not None:
            waited = request_limiter.acquire()
            if waited > 0:
                logger.info(
                    "Rate limiter waited %.2fs for %s request (%s)",
                    waited,
                    self.model_config.provider,
                    operation,
                )
        api_logger = getattr(self, "raw_api_logger", None)
        handle = None
        if api_logger is not None:
            try:
                handle = api_logger.start_request(
                    operation,
                    args,
                    kwargs,
                    getattr(self, "_raw_api_context", {}),
                    request_payload=_raw_log_request,
                )
            except Exception:
                logger.exception(
                    "Failed to record raw API request start for %s; continuing",
                    operation,
                )
        try:
            response = call(*args, **kwargs)
        except Exception as error:
            if handle is not None:
                try:
                    api_logger.record_failure(handle, error)
                except Exception:
                    logger.exception(
                        "Failed to record raw API request failure for %s; continuing",
                        operation,
                    )
            raise

        if handle is not None:
            if _raw_log_deferred:
                deferred_requests = getattr(
                    self,
                    "_deferred_raw_api_requests",
                    None,
                )
                if deferred_requests is None:
                    deferred_requests = {}
                    self._deferred_raw_api_requests = deferred_requests
                deferred_requests[id(response)] = handle
            else:
                try:
                    api_logger.record_success(handle, response)
                except Exception:
                    logger.exception(
                        "Failed to record raw API response for %s; continuing",
                        operation,
                    )
        return response

    def _record_deferred_raw_api_success(self, source: Any, response: Any) -> None:
        deferred_requests = getattr(self, "_deferred_raw_api_requests", {})
        handle = deferred_requests.pop(id(source), None)
        api_logger = getattr(self, "raw_api_logger", None)
        if handle is not None and api_logger is not None:
            try:
                api_logger.record_success(handle, response)
            except Exception:
                logger.exception(
                    "Failed to record deferred raw API response for %s; continuing",
                    handle.operation,
                )

    def _record_deferred_raw_api_failure(self, source: Any, error: BaseException) -> None:
        if source is None:
            return
        deferred_requests = getattr(self, "_deferred_raw_api_requests", {})
        handle = deferred_requests.pop(id(source), None)
        api_logger = getattr(self, "raw_api_logger", None)
        if handle is not None and api_logger is not None:
            try:
                api_logger.record_failure(handle, error)
            except Exception:
                logger.exception(
                    "Failed to record deferred raw API failure for %s; continuing",
                    handle.operation,
                )

    def get_api_key(self) -> str:
        """Read the API key named by this model configuration."""
        api_key_env = self.model_config.api_key_env
        if not api_key_env:
            raise ValueError(
                f"api_key_env must be configured for provider '{self.model_config.provider}'"
            )

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"{api_key_env} not found in environment variables")
        return api_key

    @abc.abstractmethod
    def init_client(self):
        """
        Initialize the client for the provider. Each adapter must implement this.
        Should handle API key validation and client setup.
        """
        pass
    
    @abc.abstractmethod
    def make_prediction(self, prompt: str, task_id: Optional[str] = None, test_id: Optional[str] = None, pair_index: int = None) -> Attempt:
        """
        Make a prediction with the model and return an Attempt object's answer
        
        Args:
            prompt: The prompt to send to the model
            task_id: Optional task ID to include in metadata
            test_id: Optional test ID to include in metadata
            pair_index: Optional pair index to include in metadata
        The implementation should ensure that the config name is included in the metadata.
        """
        pass

    def chat_completion(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]] = []) -> Any:
        """
        Make a raw API call to the provider and return the response
        """
        # Base implementation can raise or be empty, subclasses should override if needed
        # OpenAI-style adapters use _chat_completion internally.
        raise NotImplementedError("Subclasses must implement chat_completion if used directly.")

    @abc.abstractmethod
    def extract_json_from_response(self, input_response: str) -> List[List[int]]:
        """
        Extract JSON from various possible formats in the response
        """
        pass
