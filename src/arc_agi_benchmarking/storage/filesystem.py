"""Local filesystem storage backend."""

import logging
import os
from pathlib import Path
from typing import Optional

from arc_agi_benchmarking.storage.base import (
    StorageBackend,
    StorageReadError,
    StorageWriteError,
)

logger = logging.getLogger(__name__)


class LocalStorageBackend(StorageBackend):
    """Filesystem-based storage backend. Writes are atomic via temp file + rename."""

    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, key: str) -> Path:
        """Resolve a key to a full filesystem path within base_dir."""
        path = (self.base_dir / key).resolve()
        try:
            path.relative_to(self.base_dir.resolve())
        except ValueError:
            raise ValueError(f"Key '{key}' would escape base directory")
        return path

    def read(self, key: str) -> Optional[bytes]:
        path = self._resolve_path(key)
        if not path.exists():
            return None
        try:
            return path.read_bytes()
        except PermissionError as e:
            raise StorageReadError(f"Permission denied reading {key}: {e}")
        except OSError as e:
            raise StorageReadError(f"Failed to read {key}: {e}")

    def write(self, key: str, data: bytes) -> None:
        path = self._resolve_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            temp_path.write_bytes(data)
            os.replace(str(temp_path), str(path))
        except PermissionError as e:
            temp_path.unlink(missing_ok=True)
            raise StorageWriteError(f"Permission denied writing {key}: {e}")
        except OSError as e:
            temp_path.unlink(missing_ok=True)
            raise StorageWriteError(f"Failed to write {key}: {e}")

    def exists(self, key: str) -> bool:
        return self._resolve_path(key).exists()

    def delete(self, key: str) -> None:
        self._resolve_path(key).unlink(missing_ok=True)

    def list_keys(self, prefix: str) -> list[str]:
        prefix_path = self._resolve_path(prefix)

        if prefix_path.is_dir():
            keys = []
            for path in prefix_path.rglob("*"):
                if path.is_file():
                    keys.append(str(path.relative_to(self.base_dir)))
            return sorted(keys)

        parent = prefix_path.parent
        if not parent.exists():
            return []

        prefix_name = prefix_path.name
        keys = []
        for path in parent.iterdir():
            if path.is_file() and path.name.startswith(prefix_name):
                keys.append(str(path.relative_to(self.base_dir)))
            elif path.is_dir() and path.name.startswith(prefix_name):
                for subpath in path.rglob("*"):
                    if subpath.is_file():
                        keys.append(str(subpath.relative_to(self.base_dir)))

        return sorted(keys)

    def __repr__(self) -> str:
        return f"LocalStorageBackend(base_dir={self.base_dir})"
