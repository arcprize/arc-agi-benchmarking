"""
Lambda handlers for Step Functions orchestration.

These handlers are invoked by AWS Step Functions to orchestrate benchmark runs:
- initialize: Create run record and return task list
- handle_error: Handle task failures and update DynamoDB
- aggregate: Aggregate results from all completed tasks
- complete: Mark run as complete and publish final metrics
"""

from arc_agi_benchmarking.lambdas.initialize import handler as initialize_handler
from arc_agi_benchmarking.lambdas.handle_error import handler as handle_error_handler
from arc_agi_benchmarking.lambdas.aggregate import handler as aggregate_handler
from arc_agi_benchmarking.lambdas.complete import handler as complete_handler

__all__ = [
    "initialize_handler",
    "handle_error_handler",
    "aggregate_handler",
    "complete_handler",
]
