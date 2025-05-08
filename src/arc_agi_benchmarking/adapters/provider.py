import abc
from typing import List, Dict, Tuple, Any, Optional
import json
from datetime import datetime
from arc_agi_benchmarking.schemas import Attempt, ModelConfig
from arc_agi_benchmarking.utils.task_utils import read_models_config

class ProviderAdapter(abc.ABC):
    def __init__(self, config: str):
        """
        Initialize the provider adapter with model configuration.
        
        Args:
            config: Configuration name that identifies the model and its settings
        """
        self.config = config
        self.model_config: ModelConfig = read_models_config(config)
        
        # Verify the provider matches the adapter
        adapter_provider = self.__class__.__name__.lower().replace('adapter', '')
        if adapter_provider != self.model_config.provider:
            raise ValueError(f"Model provider mismatch. Config '{config}' is for provider '{self.model_config.provider}' but was passed to {self.__class__.__name__}")
        
        # Initialize the client
        self.client = self.init_client()

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