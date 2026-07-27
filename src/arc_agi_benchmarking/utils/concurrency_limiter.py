"""Cross-process concurrency limits for provider API calls."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import fcntl
from pathlib import Path
import re
import tempfile
from typing import IO, AsyncIterator


class ProviderConcurrencyLimiter:
    """A file-lock semaphore shared by every benchmark process for a provider."""

    def __init__(
        self,
        provider: str,
        max_concurrency: int,
        lock_root: Path | None = None,
        poll_interval: float = 0.1,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")

        safe_provider = re.sub(r"[^A-Za-z0-9_.-]+", "_", provider)
        root = lock_root or (
            Path(tempfile.gettempdir())
            / "arc-agi-benchmarking-provider-concurrency"
        )
        self.lock_dir = root / safe_provider
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        self.max_concurrency = max_concurrency
        self.poll_interval = poll_interval

        for slot_index in range(max_concurrency):
            self._slot_path(slot_index).touch(exist_ok=True)

    def _slot_path(self, slot_index: int) -> Path:
        return self.lock_dir / f"slot-{slot_index}.lock"

    def _try_acquire(self) -> IO[str] | None:
        for slot_index in range(self.max_concurrency):
            handle = self._slot_path(slot_index).open("a+")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            return handle
        return None

    async def acquire(self) -> IO[str]:
        while True:
            handle = self._try_acquire()
            if handle is not None:
                return handle
            await asyncio.sleep(self.poll_interval)

    @staticmethod
    def release(handle: IO[str]) -> None:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        handle = await self.acquire()
        try:
            yield
        finally:
            self.release(handle)
