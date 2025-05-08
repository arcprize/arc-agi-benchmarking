import asyncio
import time

class AsyncRequestRateLimiter:
    """
    An asynchronous request-based rate limiter using the token bucket algorithm.
    Uses standard floats for rate and request count tracking.
    """
    def __init__(self, rate: float, capacity: float):
        """
        Initializes the limiter.

        Args:
            rate: Allowed requests per second (e.g., 5 for 5 RPS).
            capacity: Maximum pending requests allowed (burst capacity).
        """
        if not isinstance(rate, (int, float)) or not rate > 0:
            raise ValueError("Rate must be a positive number")
        if not isinstance(capacity, (int, float)) or not capacity >= 0:
            raise ValueError("Capacity must be a non-negative number")
            
        # Use float type directly
        self._rate = float(rate) 
        self._capacity = float(capacity)
        self._available_requests = self._capacity # Start full (float)
        
        self._last_refill_time = time.monotonic() # float
        self._lock = asyncio.Lock()

    def _refill(self):
        """Calculates and adds allowed requests based on elapsed time.
        
        MUST be called when self._lock is acquired.
        Uses float for calculations.
        """
        now = time.monotonic()
        elapsed = now - self._last_refill_time # elapsed is float
        if elapsed > 0:
            # Direct float calculation
            new_requests_allowance = elapsed * self._rate
            self._available_requests = min(self._capacity, self._available_requests + new_requests_allowance)
            self._last_refill_time = now

    async def acquire(self, requests_needed: int = 1) -> None:
        """
        Acquires the specified number of requests, waiting if necessary.

        Args:
            requests_needed: The number of requests required for the operation.
                           Defaults to 1.

        Raises:
            ValueError: If requests_needed is not positive or exceeds capacity.
        """
        if not isinstance(requests_needed, int) or requests_needed <= 0:
            raise ValueError("requests_needed must be a positive integer")
            
        # Compare int requests_needed directly with float capacity
        if requests_needed > self._capacity:
            raise ValueError(
                f"Requested requests ({requests_needed}) exceeds bucket capacity "
                f"({self._capacity}) - acquisition impossible."
            )

        while True:
            async with self._lock:
                self._refill()
                
                # Compare float >= int
                if self._available_requests >= requests_needed:
                    self._available_requests -= requests_needed
                    return
                
                # Calculate wait time using float
                needed = requests_needed - self._available_requests
                # Basic check to avoid division by zero if rate is zero (shouldn't happen)
                wait_time = (needed / self._rate) if self._rate > 0 else 3600.0

            await asyncio.sleep(wait_time)

    async def __aenter__(self):
        """Async context manager entry point.
        
        Acquires 1 request allowance, waiting if necessary.
        Allows usage like: `async with limiter:`
        """
        await self.acquire(1) 
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit point.
        
        No action needed for this type of rate limiter (requests are consumed).
        """
        pass

    async def get_available_requests(self) -> float: # Return type is float
         """Returns the approximate current number of available requests as float.
         
         Performs a refill check before returning the value for accuracy.
         Uses the internal lock for consistency.
         """
         async with self._lock:
             self._refill()
             return self._available_requests 