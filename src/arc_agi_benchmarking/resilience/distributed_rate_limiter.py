"""Distributed rate limiter using DynamoDB for coordination across workers."""

import asyncio
import logging
import os
import random
import time
from decimal import Decimal
from typing import Optional

logger = logging.getLogger(__name__)


class DistributedRateLimiter:
    """Distributed token bucket rate limiter using DynamoDB.

    Uses DynamoDB conditional updates with exponential backoff to coordinate
    rate limiting across multiple workers. The conditional update ensures
    atomic token acquisition while the backoff prevents contention hot loops.

    The rate/period interface matches provider_config.yml:
        rate: 20, period: 60 means 20 requests allowed per 60 seconds.

    Note: DynamoDB doesn't support complex arithmetic in UpdateExpressions,
    so we compute refill client-side and use conditional updates for atomicity.
    The backoff mechanism handles contention when multiple workers compete.
    """

    def __init__(
        self,
        provider: str,
        rate: float,
        period: float,
        table_name: str,
        region_name: Optional[str] = None,
    ):
        """Initialize the distributed rate limiter.

        Args:
            provider: Provider name (used as partition key).
            rate: Number of requests allowed per period.
            period: Time period in seconds for the rate limit.
            table_name: DynamoDB table name for rate limit state.
            region_name: AWS region. Defaults to AWS_REGION env var.
        """
        import boto3

        if rate <= 0:
            raise ValueError("Rate must be positive")
        if period <= 0:
            raise ValueError("Period must be positive")

        self._provider = provider
        self._rate = Decimal(str(rate))
        self._period = Decimal(str(period))
        self._refill_rate = self._rate / self._period  # tokens per second
        self._region = region_name or os.environ.get("AWS_REGION", "us-west-2")

        self._dynamodb = boto3.resource("dynamodb", region_name=self._region)
        self._table = self._dynamodb.Table(table_name)

        self._ensure_item_exists()

    def _ensure_item_exists(self) -> None:
        """Create rate limit entry if it doesn't exist."""
        try:
            self._table.put_item(
                Item={
                    "provider": self._provider,
                    "tokens": self._rate,  # Start with full bucket
                    "last_update": Decimal(str(time.time())),
                },
                ConditionExpression="attribute_not_exists(provider)",
            )
            logger.debug(f"Created rate limit entry for {self._provider}")
        except self._dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
            pass  # Already exists

    async def acquire(self, tokens: int = 1, timeout: float = 60.0) -> bool:
        """Acquire tokens from the distributed bucket.

        Uses conditional updates with exponential backoff to handle contention.
        Backoff is ALWAYS applied on contention, regardless of wait_time,
        to prevent tight retry loops.

        Args:
            tokens: Number of tokens to acquire.
            timeout: Maximum time to wait in seconds.

        Returns:
            True if tokens were acquired, False if timeout reached.
        """
        if tokens <= 0:
            raise ValueError("tokens must be positive")
        if tokens > float(self._rate):
            raise ValueError(f"Requested tokens ({tokens}) exceeds rate ({self._rate})")

        start_time = time.monotonic()
        tokens_decimal = Decimal(str(tokens))
        backoff = 0.1  # Start with 100ms backoff

        while time.monotonic() - start_time < timeout:
            now = Decimal(str(time.time()))

            try:
                # Get current state
                response = await asyncio.to_thread(
                    self._table.get_item, Key={"provider": self._provider}
                )
                item = response.get("Item", {})

                last_update = item.get("last_update", now)
                current_tokens = item.get("tokens", self._rate)

                # Calculate refill (client-side since DynamoDB can't do multiplication)
                elapsed = now - last_update
                refill = elapsed * self._refill_rate
                new_tokens = min(self._rate, current_tokens + refill)

                if new_tokens >= tokens_decimal:
                    # Try atomic update with conditional check
                    await asyncio.to_thread(
                        self._table.update_item,
                        Key={"provider": self._provider},
                        UpdateExpression="SET tokens = :new_tokens, last_update = :now",
                        ConditionExpression="last_update = :old_update",
                        ExpressionAttributeValues={
                            ":new_tokens": new_tokens - tokens_decimal,
                            ":now": now,
                            ":old_update": last_update,
                        },
                    )
                    logger.debug(
                        f"Acquired {tokens_decimal} tokens for {self._provider}"
                    )
                    return True
                else:
                    # Not enough tokens, calculate wait time
                    tokens_deficit = float(tokens_decimal - new_tokens)
                    wait_time = tokens_deficit / float(self._refill_rate)

                    # Use max of wait_time and backoff to prevent tight loops
                    actual_wait = max(wait_time, backoff)

                    # Don't exceed remaining timeout
                    remaining = timeout - (time.monotonic() - start_time)
                    actual_wait = min(actual_wait, remaining)

                    if actual_wait > 0:
                        logger.debug(
                            f"Waiting {actual_wait:.2f}s for tokens ({self._provider})"
                        )
                        await asyncio.sleep(actual_wait)

                    # Exponential backoff with jitter
                    backoff = min(backoff * 2 * (0.5 + random.random()), 5.0)

            except (
                self._dynamodb.meta.client.exceptions.ConditionalCheckFailedException
            ):
                # Contention: another worker modified the state
                # ALWAYS apply backoff here to prevent hot loops
                remaining = timeout - (time.monotonic() - start_time)
                actual_wait = min(backoff, remaining)

                if actual_wait > 0:
                    logger.debug(
                        f"Contention for {self._provider}, backing off {actual_wait:.2f}s"
                    )
                    await asyncio.sleep(actual_wait)

                # Exponential backoff with jitter (max 5s)
                backoff = min(backoff * 2 * (0.5 + random.random()), 5.0)

        logger.warning(f"Rate limit timeout for {self._provider} after {timeout}s")
        return False

    async def get_available_tokens(self) -> float:
        """Get approximate available tokens (for monitoring)."""
        try:
            response = await asyncio.to_thread(
                self._table.get_item, Key={"provider": self._provider}
            )
            item = response.get("Item", {})

            last_update = item.get("last_update", Decimal(str(time.time())))
            current_tokens = item.get("tokens", self._rate)

            # Calculate current tokens with refill
            now = Decimal(str(time.time()))
            elapsed = now - last_update
            refill = elapsed * self._refill_rate
            available = min(self._rate, current_tokens + refill)

            return float(available)
        except Exception:
            return 0.0

    @property
    def rate(self) -> float:
        """Get the rate limit (requests per period)."""
        return float(self._rate)

    @property
    def period(self) -> float:
        """Get the period in seconds."""
        return float(self._period)

    async def __aenter__(self):
        """Async context manager - acquires 1 token."""
        await self.acquire(1)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass
