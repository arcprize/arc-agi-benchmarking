"""S3 storage backend for AWS deployments."""

import logging
from typing import Optional

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    raise ImportError(
        "boto3 is required for S3StorageBackend. "
        "Install it with: pip install boto3"
    )

from arc_agi_benchmarking.storage.base import (
    StorageBackend,
    StorageReadError,
    StorageWriteError,
)

logger = logging.getLogger(__name__)


class S3StorageBackend(StorageBackend):
    """S3-based storage backend. S3 PutObject is atomic."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region_name: Optional[str] = None,
    ):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.s3 = boto3.client("s3", region_name=region_name)

    def _full_key(self, key: str) -> str:
        if self.prefix:
            return f"{self.prefix}/{key}"
        return key

    def _strip_prefix(self, full_key: str) -> str:
        if self.prefix and full_key.startswith(f"{self.prefix}/"):
            return full_key[len(self.prefix) + 1 :]
        return full_key

    def read(self, key: str) -> Optional[bytes]:
        full_key = self._full_key(key)
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=full_key)
            return response["Body"].read()
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "NoSuchKey":
                return None
            logger.error(f"S3 read error for {key}: {e}")
            raise StorageReadError(f"Failed to read {key} from S3: {e}")

    def write(self, key: str, data: bytes) -> None:
        full_key = self._full_key(key)
        try:
            self.s3.put_object(Bucket=self.bucket, Key=full_key, Body=data)
        except ClientError as e:
            logger.error(f"S3 write error for {key}: {e}")
            raise StorageWriteError(f"Failed to write {key} to S3: {e}")

    def exists(self, key: str) -> bool:
        full_key = self._full_key(key)
        try:
            self.s3.head_object(Bucket=self.bucket, Key=full_key)
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "404":
                return False
            logger.warning(f"S3 head_object error for {key}: {e}")
            return False

    def delete(self, key: str) -> None:
        full_key = self._full_key(key)
        try:
            self.s3.delete_object(Bucket=self.bucket, Key=full_key)
        except ClientError as e:
            logger.warning(f"S3 delete error for {key}: {e}")

    def list_keys(self, prefix: str) -> list[str]:
        full_prefix = self._full_key(prefix)
        keys = []
        paginator = self.s3.get_paginator("list_objects_v2")
        try:
            for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
                for obj in page.get("Contents", []):
                    keys.append(self._strip_prefix(obj["Key"]))
        except ClientError as e:
            logger.error(f"S3 list error for prefix {prefix}: {e}")
            return []
        return sorted(keys)

    def __repr__(self) -> str:
        prefix_str = f", prefix={self.prefix}" if self.prefix else ""
        return f"S3StorageBackend(bucket={self.bucket}{prefix_str})"
