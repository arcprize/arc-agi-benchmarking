"""Tests for storage backends."""

import os
import tempfile
from pathlib import Path

import pytest

from arc_agi_benchmarking.storage.base import (
    StorageBackend,
    StorageReadError,
    StorageWriteError,
)
from arc_agi_benchmarking.storage.filesystem import LocalStorageBackend


class TestLocalStorageBackend:
    """Tests for LocalStorageBackend."""

    @pytest.fixture
    def storage(self, tmp_path: Path) -> LocalStorageBackend:
        """Create a LocalStorageBackend with a temporary directory."""
        return LocalStorageBackend(tmp_path)

    def test_init_creates_directory(self, tmp_path: Path):
        """Test that initialization creates the base directory."""
        new_dir = tmp_path / "new_storage_dir"
        assert not new_dir.exists()

        storage = LocalStorageBackend(new_dir)

        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_write_and_read(self, storage: LocalStorageBackend):
        """Test basic write and read operations."""
        data = b"hello world"
        storage.write("test.txt", data)

        result = storage.read("test.txt")
        assert result == data

    def test_write_and_read_text(self, storage: LocalStorageBackend):
        """Test text convenience methods."""
        text = "hello world"
        storage.write_text("test.txt", text)

        result = storage.read_text("test.txt")
        assert result == text

    def test_read_nonexistent_returns_none(self, storage: LocalStorageBackend):
        """Test that reading a nonexistent key returns None."""
        result = storage.read("nonexistent.txt")
        assert result is None

    def test_write_creates_parent_directories(self, storage: LocalStorageBackend):
        """Test that write creates necessary parent directories."""
        data = b"nested data"
        storage.write("deeply/nested/path/file.txt", data)

        result = storage.read("deeply/nested/path/file.txt")
        assert result == data

    def test_write_is_atomic(self, storage: LocalStorageBackend, tmp_path: Path):
        """Test that writes are atomic (no partial writes visible)."""
        key = "atomic_test.txt"
        data1 = b"original data"
        data2 = b"new data that is longer"

        # Write initial data
        storage.write(key, data1)

        # Write new data - should be atomic
        storage.write(key, data2)

        # Verify new data is complete
        result = storage.read(key)
        assert result == data2

        # Verify no temp files left behind
        files = list(tmp_path.rglob("*.tmp"))
        assert len(files) == 0

    def test_exists(self, storage: LocalStorageBackend):
        """Test exists method."""
        assert not storage.exists("test.txt")

        storage.write("test.txt", b"data")

        assert storage.exists("test.txt")

    def test_delete(self, storage: LocalStorageBackend):
        """Test delete method."""
        storage.write("test.txt", b"data")
        assert storage.exists("test.txt")

        storage.delete("test.txt")

        assert not storage.exists("test.txt")

    def test_delete_nonexistent_does_not_raise(self, storage: LocalStorageBackend):
        """Test that deleting a nonexistent key doesn't raise."""
        # Should not raise
        storage.delete("nonexistent.txt")

    def test_list_keys_in_directory(self, storage: LocalStorageBackend):
        """Test listing keys in a directory."""
        storage.write("dir/file1.txt", b"1")
        storage.write("dir/file2.txt", b"2")
        storage.write("dir/subdir/file3.txt", b"3")
        storage.write("other/file4.txt", b"4")

        keys = storage.list_keys("dir")

        assert sorted(keys) == [
            "dir/file1.txt",
            "dir/file2.txt",
            "dir/subdir/file3.txt",
        ]

    def test_list_keys_with_prefix(self, storage: LocalStorageBackend):
        """Test listing keys with a filename prefix."""
        storage.write("checkpoint_001.json", b"1")
        storage.write("checkpoint_002.json", b"2")
        storage.write("submission_001.json", b"3")

        keys = storage.list_keys("checkpoint")

        assert sorted(keys) == ["checkpoint_001.json", "checkpoint_002.json"]

    def test_list_keys_empty(self, storage: LocalStorageBackend):
        """Test listing keys when no matches."""
        keys = storage.list_keys("nonexistent")
        assert keys == []

    def test_path_traversal_blocked(self, storage: LocalStorageBackend):
        """Test that path traversal attacks are blocked."""
        with pytest.raises(ValueError, match="escape base directory"):
            storage.write("../../../etc/passwd", b"malicious")

        with pytest.raises(ValueError, match="escape base directory"):
            storage.read("../../../etc/passwd")

    def test_overwrite_existing_file(self, storage: LocalStorageBackend):
        """Test overwriting an existing file."""
        storage.write("test.txt", b"original")
        storage.write("test.txt", b"updated")

        result = storage.read("test.txt")
        assert result == b"updated"

    def test_empty_data(self, storage: LocalStorageBackend):
        """Test writing and reading empty data."""
        storage.write("empty.txt", b"")

        result = storage.read("empty.txt")
        assert result == b""

    def test_binary_data(self, storage: LocalStorageBackend):
        """Test writing and reading binary data."""
        data = bytes(range(256))
        storage.write("binary.dat", data)

        result = storage.read("binary.dat")
        assert result == data

    def test_unicode_in_text(self, storage: LocalStorageBackend):
        """Test unicode characters in text mode."""
        text = "Hello \u4e16\u754c \U0001f600"  # "Hello 世界 😀"
        storage.write_text("unicode.txt", text)

        result = storage.read_text("unicode.txt")
        assert result == text

    def test_repr(self, storage: LocalStorageBackend, tmp_path: Path):
        """Test string representation."""
        repr_str = repr(storage)
        assert "LocalStorageBackend" in repr_str
        assert str(tmp_path) in repr_str


class TestS3StorageBackend:
    """Tests for S3StorageBackend using moto for mocking."""

    @pytest.fixture
    def s3_storage(self):
        """Create an S3StorageBackend with mocked S3."""
        try:
            import boto3
            from moto import mock_aws
        except ImportError:
            pytest.skip("boto3 and moto required for S3 tests")

        with mock_aws():
            # Create the bucket
            s3 = boto3.client("s3", region_name="us-east-1")
            s3.create_bucket(Bucket="test-bucket")

            from arc_agi_benchmarking.storage.s3 import S3StorageBackend

            storage = S3StorageBackend(
                bucket="test-bucket",
                prefix="test-prefix",
                region_name="us-east-1",
            )
            yield storage

    def test_write_and_read(self, s3_storage):
        """Test basic write and read operations."""
        data = b"hello world"
        s3_storage.write("test.txt", data)

        result = s3_storage.read("test.txt")
        assert result == data

    def test_read_nonexistent_returns_none(self, s3_storage):
        """Test that reading a nonexistent key returns None."""
        result = s3_storage.read("nonexistent.txt")
        assert result is None

    def test_exists(self, s3_storage):
        """Test exists method."""
        assert not s3_storage.exists("test.txt")

        s3_storage.write("test.txt", b"data")

        assert s3_storage.exists("test.txt")

    def test_delete(self, s3_storage):
        """Test delete method."""
        s3_storage.write("test.txt", b"data")
        assert s3_storage.exists("test.txt")

        s3_storage.delete("test.txt")

        assert not s3_storage.exists("test.txt")

    def test_delete_nonexistent_does_not_raise(self, s3_storage):
        """Test that deleting a nonexistent key doesn't raise."""
        # Should not raise
        s3_storage.delete("nonexistent.txt")

    def test_list_keys(self, s3_storage):
        """Test listing keys with a prefix."""
        s3_storage.write("dir/file1.txt", b"1")
        s3_storage.write("dir/file2.txt", b"2")
        s3_storage.write("other/file3.txt", b"3")

        keys = s3_storage.list_keys("dir/")

        assert sorted(keys) == ["dir/file1.txt", "dir/file2.txt"]

    def test_prefix_applied_correctly(self, s3_storage):
        """Test that the storage prefix is applied correctly."""
        # Write through our storage
        s3_storage.write("myfile.txt", b"content")

        # Verify the full S3 key includes our prefix
        import boto3

        s3 = boto3.client("s3", region_name="us-east-1")
        response = s3.get_object(Bucket="test-bucket", Key="test-prefix/myfile.txt")
        assert response["Body"].read() == b"content"

    def test_repr(self, s3_storage):
        """Test string representation."""
        repr_str = repr(s3_storage)
        assert "S3StorageBackend" in repr_str
        assert "test-bucket" in repr_str
        assert "test-prefix" in repr_str


class TestStorageBackendInterface:
    """Tests to verify the interface contract."""

    def test_local_implements_interface(self, tmp_path: Path):
        """Test that LocalStorageBackend implements StorageBackend."""
        storage = LocalStorageBackend(tmp_path)
        assert isinstance(storage, StorageBackend)

    def test_s3_implements_interface(self):
        """Test that S3StorageBackend implements StorageBackend."""
        try:
            from moto import mock_aws
            import boto3
        except ImportError:
            pytest.skip("boto3 and moto required for S3 tests")

        with mock_aws():
            s3 = boto3.client("s3", region_name="us-east-1")
            s3.create_bucket(Bucket="test-bucket")

            from arc_agi_benchmarking.storage.s3 import S3StorageBackend

            storage = S3StorageBackend(bucket="test-bucket")
            assert isinstance(storage, StorageBackend)
