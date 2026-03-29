"""
ARKEN — Storage Service v3.0
Local-first with optional S3 upgrade.
USE_S3=false (default) stores everything to /tmp/arken_local_storage.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)

LOCAL_DIR = Path("/tmp/arken_local_storage")


class StorageService:
    """
    Unified storage: local filesystem (default) or AWS S3.
    Never raises on missing credentials — gracefully degrades.
    """

    def __init__(self):
        self._s3 = None
        self._use_s3 = getattr(settings, "USE_S3", False)

    def _get_s3(self):
        if self._s3 is None:
            try:
                import boto3
                key = settings.AWS_ACCESS_KEY_ID
                secret = None
                if settings.AWS_SECRET_ACCESS_KEY:
                    try:
                        secret = settings.AWS_SECRET_ACCESS_KEY.get_secret_value()
                    except Exception:
                        secret = str(settings.AWS_SECRET_ACCESS_KEY)
                self._s3 = boto3.client(
                    "s3",
                    region_name=settings.AWS_REGION,
                    aws_access_key_id=key,
                    aws_secret_access_key=secret,
                )
            except Exception as e:
                logger.warning(f"[Storage] S3 init failed: {e} — using local")
                self._use_s3 = False
        return self._s3

    # ── Public API ──────────────────────────────────────────────────────────

    async def upload_bytes(
        self,
        data: bytes,
        key: str,
        content_type: str = "application/octet-stream",
        bucket: Optional[str] = None,
    ) -> str:
        if self._use_s3:
            try:
                return self._s3_upload(data, key, content_type, bucket)
            except Exception as e:
                logger.warning(f"[Storage] S3 upload failed: {e} — falling back to local")

        return self._local_write(data, key)

    async def download_bytes(self, key: str, bucket: Optional[str] = None) -> bytes:
        if self._use_s3:
            try:
                return self._s3_download(key, bucket)
            except Exception as e:
                logger.warning(f"[Storage] S3 download failed: {e} — trying local")

        return self._local_read(key)

    async def delete(self, key: str, bucket: Optional[str] = None):
        if self._use_s3:
            try:
                s3 = self._get_s3()
                s3.delete_object(Bucket=bucket or settings.S3_BUCKET_UPLOADS, Key=key)
                return
            except Exception:
                pass
        local_path = LOCAL_DIR / key
        if local_path.exists():
            local_path.unlink()

    async def exists(self, key: str, bucket: Optional[str] = None) -> bool:
        if self._use_s3:
            try:
                s3 = self._get_s3()
                s3.head_object(Bucket=bucket or settings.S3_BUCKET_UPLOADS, Key=key)
                return True
            except Exception:
                pass
        return (LOCAL_DIR / key).exists()

    def get_presigned_url(self, key: str, bucket: str, expiry: int = 3600) -> str:
        if self._use_s3:
            try:
                s3 = self._get_s3()
                return s3.generate_presigned_url(
                    "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expiry
                )
            except Exception:
                pass
        return f"/api/v1/artifacts/local/{key}"

    # ── Internals ────────────────────────────────────────────────────────────

    def _s3_upload(self, data: bytes, key: str, content_type: str, bucket: Optional[str]) -> str:
        bucket = bucket or settings.S3_BUCKET_UPLOADS
        s3 = self._get_s3()
        s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
        if settings.CDN_BASE_URL:
            return f"{settings.CDN_BASE_URL}/{key}"
        return f"https://{bucket}.s3.{settings.AWS_REGION}.amazonaws.com/{key}"

    def _s3_download(self, key: str, bucket: Optional[str]) -> bytes:
        bucket = bucket or settings.S3_BUCKET_UPLOADS
        s3 = self._get_s3()
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()

    def _local_write(self, data: bytes, key: str) -> str:
        path = LOCAL_DIR / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        logger.debug(f"[Storage] Local write: {path}")
        return f"local://{key}"

    def _local_read(self, key: str) -> bytes:
        # Strip local:// prefix if present
        clean = key.replace("local://", "")
        path = LOCAL_DIR / clean
        if path.exists():
            return path.read_bytes()
        raise FileNotFoundError(f"Local file not found: {path}")


s3_service = StorageService()
