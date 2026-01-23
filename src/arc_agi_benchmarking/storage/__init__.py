"""Storage abstraction layer for checkpoints and submissions."""

from arc_agi_benchmarking.storage.base import StorageBackend
from arc_agi_benchmarking.storage.filesystem import LocalStorageBackend

__all__ = [
    "StorageBackend",
    "LocalStorageBackend",
]

try:
    from arc_agi_benchmarking.storage.s3 import S3StorageBackend
    __all__.append("S3StorageBackend")
except ImportError:
    pass
