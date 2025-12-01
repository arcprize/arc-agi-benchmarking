import json
import random
from datetime import datetime, timezone
from typing import Any, List, Optional

from arc_agi_benchmarking.schemas import (
    Attempt,
    AttemptMetadata,
    Choice,
    CompletionTokensDetails,
    Cost,
    Message,
    Usage,
)
from .provider import ProviderAdapter


class RandomAdapter(ProviderAdapter):
    """
    A local, dependency-free adapter that generates a random output grid matching the
    test input shape. Intended for quickstarts and smoke tests without API keys.
    """

    def init_client(self):
        # No external client needed.
        return None

    def _extract_test_input_grid(self, prompt: str) -> List[List[int]]:
        """
        Parse the test input grid from the prompt template.
        Falls back to a 1x1 zero grid if parsing fails.
        """
        try:
            start = prompt.index("--Test Input--") + len("--Test Input--")
            end = prompt.index("--End of Test Input--", start)
            grid_str = prompt[start:end].strip()
            return json.loads(grid_str)
        except Exception:
            return [[0]]

    def _random_grid(self, shape_like: List[List[int]]) -> List[List[int]]:
        rows = len(shape_like)
        cols = len(shape_like[0]) if rows else 1
        return [
            [random.randint(0, 9) for _ in range(cols)]
            for _ in range(rows)
        ]

    def make_prediction(
        self,
        prompt: str,
        task_id: Optional[str] = None,
        test_id: Optional[str] = None,
        pair_index: int = None,
    ) -> Attempt:
        start_time = datetime.now(timezone.utc)

        test_grid = self._extract_test_input_grid(prompt)
        answer_grid = self._random_grid(test_grid)

        end_time = datetime.now(timezone.utc)

        choices = [
            Choice(
                index=0,
                message=Message(role="user", content="ARC task prompt omitted for random solver"),
            ),
            Choice(
                index=1,
                message=Message(role="assistant", content=json.dumps(answer_grid)),
            ),
        ]

        metadata = AttemptMetadata(
            model=self.model_config.model_name,
            provider=self.model_config.provider,
            start_timestamp=start_time,
            end_timestamp=end_time,
            choices=choices,
            kwargs=self.model_config.kwargs,
            usage=Usage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                completion_tokens_details=CompletionTokensDetails(
                    reasoning_tokens=0,
                    accepted_prediction_tokens=0,
                    rejected_prediction_tokens=0,
                ),
            ),
            cost=Cost(
                prompt_cost=0.0,
                completion_cost=0.0,
                reasoning_cost=None,
                total_cost=0.0,
            ),
            task_id=task_id,
            pair_index=pair_index,
            test_id=test_id,
        )

        return Attempt(metadata=metadata, answer=answer_grid)

    def extract_json_from_response(self, input_response: str) -> List[List[int]]:
        # Not used for this adapter because we return structured answers directly.
        try:
            return json.loads(input_response)
        except Exception:
            return [[0]]
