from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import time
from typing import Any, Iterator, Optional
from urllib.parse import urlparse

import boto3
from botocore.client import BaseClient
from botocore.config import Config
from botocore.exceptions import (
    ClientError,
    ConnectionClosedError,
    HTTPClientError,
    ReadTimeoutError,
    ResponseStreamingError,
)


@dataclass(frozen=True)
class S3Location:
    bucket: str
    key: str


def parse_s3_uri(uri: str) -> S3Location:
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"expected s3:// URI, got: {uri}")

    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    if not bucket or not key:
        raise ValueError(f"invalid S3 URI: {uri}")

    return S3Location(bucket=bucket, key=key)


@lru_cache(maxsize=8)
def get_s3_client(max_pool_connections: int = 100) -> BaseClient:
    config = Config(
        max_pool_connections=max_pool_connections,
        retries={
            "max_attempts": 10,
            "mode": "standard",
        },
    )
    return boto3.client("s3", config=config)


def _is_retryable_s3_read_error(exc: BaseException) -> bool:
    return isinstance(
        exc,
        (
            ResponseStreamingError,
            ReadTimeoutError,
            HTTPClientError,
            ConnectionClosedError,
        ),
    )


def read_s3_bytes(
    uri: str,
    *,
    s3_client: Optional[BaseClient] = None,
    max_attempts: int = 5,
    initial_backoff_seconds: float = 0.1,
) -> bytes:
    loc = parse_s3_uri(uri)
    s3 = s3_client or get_s3_client()

    last_exc: BaseException | None = None

    for attempt in range(1, max_attempts + 1):
        body = None
        try:
            resp = s3.get_object(Bucket=loc.bucket, Key=loc.key)
            body = resp["Body"]
            data = body.read()
            return data

        except Exception as exc:
            last_exc = exc

            if attempt >= max_attempts or not _is_retryable_s3_read_error(exc):
                raise RuntimeError(
                    f"failed to read S3 object uri={uri} bucket={loc.bucket} key={loc.key} "
                    f"after {attempt} attempt(s)"
                ) from exc

            time.sleep(initial_backoff_seconds * (2 ** (attempt - 1)))

        finally:
            if body is not None:
                try:
                    body.close()
                except Exception:
                    pass

    raise RuntimeError(
        f"failed to read S3 object uri={uri} bucket={loc.bucket} key={loc.key}"
    ) from last_exc


def iter_s3_prefix(prefix_uri: str) -> Iterator[str]:
    loc = parse_s3_uri(prefix_uri)
    s3 = get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")

    try:
        for page in paginator.paginate(Bucket=loc.bucket, Prefix=loc.key):
            for item in page.get("Contents", []):
                key = item["Key"]
                if key.endswith("/"):
                    continue
                yield f"s3://{loc.bucket}/{key}"
    except Exception as exc:
        raise RuntimeError(
            f"failed to list S3 prefix uri={prefix_uri} bucket={loc.bucket} prefix={loc.key}"
        ) from exc


def s3_key_exists(uri: str) -> bool:
    loc = parse_s3_uri(uri)
    s3 = get_s3_client()

    try:
        s3.head_object(Bucket=loc.bucket, Key=loc.key)
        return True
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "")
        if error_code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def read_s3_json(uri: str) -> dict[str, Any] | list[Any]:
    try:
        raw = read_s3_bytes(uri)
        return json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"failed to read JSON from S3 uri={uri}") from exc


def write_s3_json(uri: str, payload: Any) -> None:
    loc = parse_s3_uri(uri)
    s3 = get_s3_client()

    try:
        s3.put_object(
            Bucket=loc.bucket,
            Key=loc.key,
            Body=json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
        )
    except Exception as exc:
        raise RuntimeError(
            f"failed to write JSON to S3 uri={uri} bucket={loc.bucket} key={loc.key}"
        ) from exc