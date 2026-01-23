"""Abstract base class for storage backends."""

from abc import ABC, abstractmethod
from typing import Optional


class StorageBackend(ABC):
    """Abstract storage backend for checkpoints and submissions.

    Keys are path-like strings (e.g., "checkpoints/task_123.json").
    """

    @abstractmethod
    def read(self, key: str) -> Optional[bytes]:
        """Read data from storage. Returns None if key doesn't exist."""
        pass

    @abstractmethod
    def write(self, key: str, data: bytes) -> None:
        """Write data to storage atomically."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a key exists in storage."""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a key from storage. No-op if key doesn't exist."""
        pass

    @abstractmethod
    def list_keys(self, prefix: str) -> list[str]:
        """List all keys with a given prefix."""
        pass

    def read_text(self, key: str, encoding: str = "utf-8") -> Optional[str]:
        """Read data as text."""
        data = self.read(key)
        if data is None:
            return None
        return data.decode(encoding)

    def write_text(self, key: str, text: str, encoding: str = "utf-8") -> None:
        """Write text data."""
        self.write(key, text.encode(encoding))


class StorageError(Exception):
    """Base exception for storage errors."""

    pass


class StorageWriteError(StorageError):
    """Raised when a write operation fails."""

    pass


class StorageReadError(StorageError):
    """Raised when a read operation fails unexpectedly."""

    pass
