from __future__ import annotations

import io
from typing import Any

import torch

from batchflow.common.payload_formats import PayloadFormat


def decode_payload(
    payload: bytes,
    *,
    payload_format: str,
) -> dict[str, Any]:
    try:
        fmt = PayloadFormat(payload_format)
    except ValueError as exc:
        raise ValueError(f"unknown payload_format={payload_format!r}") from exc

    if fmt != PayloadFormat.TORCH_BATCH:
        raise ValueError(
            f"PyTorch integration expected payload_format="
            f"{PayloadFormat.TORCH_BATCH.value!r}, got {payload_format!r}"
        )

    return decode_torch_batch(payload)


def decode_torch_batch(payload: bytes) -> dict[str, Any]:
    buffer = io.BytesIO(payload)
    decoded = torch.load(buffer, map_location="cpu")

    if not isinstance(decoded, dict):
        raise TypeError(
            f"expected torch batch payload to decode to dict, "
            f"got {type(decoded).__name__}"
        )
    return decoded

def pin_memory_batch(obj: Any) -> Any:
    if not torch.cuda.is_available():
        return obj

    return _pin_memory_batch(obj)


def _pin_memory_batch(obj: Any) -> Any:
    if torch.is_tensor(obj):
        try:
            return obj.pin_memory()
        except RuntimeError:
            return obj

    if isinstance(obj, dict):
        return {key: _pin_memory_batch(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [_pin_memory_batch(value) for value in obj]

    if isinstance(obj, tuple):
        return tuple(_pin_memory_batch(value) for value in obj)

    return obj