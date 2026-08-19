from __future__ import annotations

import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

import torch
from PIL import Image

from batchflow.common.payload_formats import PayloadFormat
from batchflow.common.s3io import read_s3_bytes
from batchflow.common.transforms import build_image_transform, build_text_transform
from batchflow.proto import batchflow_pb2


@dataclass(frozen=True)
class MaterializedPayload:
    payload: bytes
    payload_format: str


class BatchMaterializer:
    def __init__(
        self,
        *,
        s3_fetch_threads: int,
        decode_threads: int,
    ) -> None:
        self.s3_fetch_threads = s3_fetch_threads
        self.decode_threads = decode_threads
        self.s3_client = None
        self._text_transforms: dict[str, Callable[[str], torch.Tensor]] = {}

    def materialize(
        self,
        batch: batchflow_pb2.Batch,
        *,
        payload_format: str,
        dataset_format: str,
    ) -> MaterializedPayload:
        requested_format = PayloadFormat(payload_format)

        if dataset_format == "coco_retrieval":
            if requested_format != PayloadFormat.TORCH_BATCH:
                raise ValueError(
                    "COCO retrieval batches require payload_format="
                    f"{PayloadFormat.TORCH_BATCH.value!r}, got {payload_format!r}"
                )

            return MaterializedPayload(
                payload=self._materialize_coco_retrieval_torch_batch(batch),
                payload_format=PayloadFormat.TORCH_BATCH.value,
            )

        if _is_s3_image_batch(batch):
            if requested_format != PayloadFormat.TORCH_BATCH:
                raise ValueError(
                    f"S3 image batches only support payload_format="
                    f"{PayloadFormat.TORCH_BATCH.value!r}, got {payload_format!r}"
                )

            return MaterializedPayload(
                payload=self._materialize_image_torch_batch(batch),
                payload_format=PayloadFormat.TORCH_BATCH.value,
            )

        if _is_synthetic_torch_batch(batch):
            if requested_format != PayloadFormat.TORCH_BATCH:
                raise ValueError(
                    "Synthetic PyTorch batches require payload_format="
                    f"{PayloadFormat.TORCH_BATCH.value!r}, got {payload_format!r}"
                )

            return MaterializedPayload(
                payload=self._materialize_synthetic_torch_batch(batch),
                payload_format=PayloadFormat.TORCH_BATCH.value,
            )

        sample_uris = [sample.source_uri for sample in batch.samples[:3]]

        raise NotImplementedError(
            "unsupported batch type or payload format; "
            f"dataset_format={dataset_format!r} "
            f"payload_format={payload_format!r} "
            f"sample_uris={sample_uris}"
        )

    def _fetch_one_sample(
        self,
        sample: batchflow_pb2.Sample,
    ) -> tuple[batchflow_pb2.Sample, bytes, float]:
        io_start = time.perf_counter()
        raw = read_s3_bytes(sample.source_uri, s3_client=self.s3_client)
        worker_io_time_sec = time.perf_counter() - io_start

        return sample, raw, worker_io_time_sec

    def _decode_transform_one_sample(
        self,
        item: tuple[int, batchflow_pb2.Sample, bytes, float],
        *,
        transform: Callable[[Image.Image], torch.Tensor],
    ) -> tuple[int, dict[str, Any]]:
        index, sample, raw, worker_io_time_sec = item

        decode_start = time.perf_counter()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        worker_decode_time_sec = time.perf_counter() - decode_start

        transform_start = time.perf_counter()
        image_tensor = transform(image)
        worker_transform_time_sec = time.perf_counter() - transform_start

        return index, {
            "image_tensor": image_tensor,
            "label": int(sample.label) if sample.has_label else -1,
            "sample_id": sample.sample_id,
            "class_name": sample.class_name,
            "source_uri": sample.source_uri,
            "worker_io_time_sec": worker_io_time_sec,
            "worker_decode_time_sec": worker_decode_time_sec,
            "worker_transform_time_sec": worker_transform_time_sec,
        }

    def _materialize_image_torch_batch(
        self,
        batch: batchflow_pb2.Batch,
    ) -> bytes:
        transform = build_image_transform(batch.transform_name or None)

        fetched = self._fetch_samples(batch.samples)

        indexed_fetched = [
            (index, sample, raw, worker_io_time_sec)
            for index, (sample, raw, worker_io_time_sec) in enumerate(fetched)
        ]

        decoded_transformed = self._decode_transform_samples(
            indexed_fetched,
            transform=transform,
        )

        decoded_transformed.sort(key=lambda item: item[0])

        images: list[torch.Tensor] = []
        labels: list[int] = []
        sample_ids: list[str] = []
        class_names: list[str] = []
        source_uris: list[str] = []
        worker_io_times: list[float] = []
        worker_decode_times: list[float] = []
        worker_transform_times: list[float] = []

        for _, result in decoded_transformed:
            images.append(result["image_tensor"])
            labels.append(result["label"])
            sample_ids.append(result["sample_id"])
            class_names.append(result["class_name"])
            source_uris.append(result["source_uri"])
            worker_io_times.append(result["worker_io_time_sec"])
            worker_decode_times.append(result["worker_decode_time_sec"])
            worker_transform_times.append(result["worker_transform_time_sec"])

        stack_start = time.perf_counter()
        image_batch = torch.stack(images, dim=0)
        label_batch = torch.tensor(labels, dtype=torch.long)
        worker_stack_time_sec = time.perf_counter() - stack_start

        batch_payload = {
            "image": image_batch,
            "label": label_batch,
            "sample_id": sample_ids,
            "class_name": class_names,
            "source_uri": source_uris,
            "worker_io_time_sec": torch.tensor(worker_io_times, dtype=torch.float32),
            "worker_decode_time_sec": torch.tensor(
                worker_decode_times,
                dtype=torch.float32,
            ),
            "worker_transform_time_sec": torch.tensor(
                worker_transform_times,
                dtype=torch.float32,
            ),
            "worker_stack_time_sec": float(worker_stack_time_sec),
            "worker_serialize_time_sec": 0.0,
            "batch_size": len(images),
            "batch_id": batch.batch_id,
            "dataset_id": batch.dataset_id,
            "batch_index": int(batch.batch_index),
            "epoch": int(batch.epoch),
            "transform_name": batch.transform_name,
            "payload_format": PayloadFormat.TORCH_BATCH.value,
        }

        payload, worker_serialize_time_sec = _torch_serialize_with_timing(
            batch_payload,
        )

        batch_payload["worker_serialize_time_sec"] = float(worker_serialize_time_sec)

        final_payload, _ = _torch_serialize_with_timing(batch_payload)
        return final_payload

    def _materialize_coco_retrieval_torch_batch(
        self,
        batch: batchflow_pb2.Batch,
    ) -> bytes:
        image_transform = build_image_transform(batch.transform_name or "albef_train_384")
        metadata = _batch_metadata_dict(batch)
        text_transform_name = metadata.get("text_transform_name", "albef_text_30")
        text_transform = self._text_transforms.get(text_transform_name)

        if text_transform is None:
            text_transform = build_text_transform(text_transform_name)
            self._text_transforms[text_transform_name] = text_transform

        fetched = self._fetch_samples(batch.samples)
        indexed = [
            (index, sample, raw, worker_io_time_sec)
            for index, (sample, raw, worker_io_time_sec) in enumerate(fetched)
        ]

        def prepare(item):
            index, sample, raw, worker_io_time_sec = item
            decode_start = time.perf_counter()
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            worker_decode_time_sec = time.perf_counter() - decode_start

            if not sample.has_label:
                raise ValueError(
                    f"COCO retrieval sample {sample.sample_id!r} has no image index"
                )
            if not sample.class_name:
                raise ValueError(
                    f"COCO retrieval sample {sample.sample_id!r} has no caption"
                )

            transform_start = time.perf_counter()
            image_tensor = image_transform(image)
            text_tensor = text_transform(sample.class_name)
            worker_transform_time_sec = time.perf_counter() - transform_start

            return index, {
                "image": image_tensor,
                "text": text_tensor,
                "text_atts": (text_tensor != 0).long(),
                "idx": int(sample.label),
                "sample_id": sample.sample_id,
                "source_uri": sample.source_uri,
                "worker_io_time_sec": worker_io_time_sec,
                "worker_decode_time_sec": worker_decode_time_sec,
                "worker_transform_time_sec": worker_transform_time_sec,
            }

        if self.decode_threads <= 1:
            prepared = [prepare(item) for item in indexed]
        else:
            with ThreadPoolExecutor(max_workers=self.decode_threads) as executor:
                prepared = list(executor.map(prepare, indexed))

        prepared.sort(key=lambda item: item[0])
        results = [result for _, result in prepared]

        stack_start = time.perf_counter()
        image_batch = torch.stack([item["image"] for item in results], dim=0)
        text_batch = torch.stack([item["text"] for item in results], dim=0)
        text_atts_batch = torch.stack([item["text_atts"] for item in results], dim=0)
        idx_batch = torch.tensor([item["idx"] for item in results], dtype=torch.long)
        worker_stack_time_sec = time.perf_counter() - stack_start

        batch_payload = {
            "image": image_batch,
            "text": text_batch,
            "text_atts": text_atts_batch,
            "idx": idx_batch,
            "sample_id": [item["sample_id"] for item in results],
            "source_uri": [item["source_uri"] for item in results],
            "worker_io_time_sec": torch.tensor(
                [item["worker_io_time_sec"] for item in results], dtype=torch.float32
            ),
            "worker_decode_time_sec": torch.tensor(
                [item["worker_decode_time_sec"] for item in results], dtype=torch.float32
            ),
            "worker_transform_time_sec": torch.tensor(
                [item["worker_transform_time_sec"] for item in results], dtype=torch.float32
            ),
            "worker_stack_time_sec": float(worker_stack_time_sec),
            "worker_serialize_time_sec": 0.0,
            "batch_size": len(results),
            "batch_id": batch.batch_id,
            "dataset_id": batch.dataset_id,
            "batch_index": int(batch.batch_index),
            "epoch": int(batch.epoch),
            "transform_name": batch.transform_name,
            "text_transform_name": text_transform_name,
            "payload_format": PayloadFormat.TORCH_BATCH.value,
        }

        _, worker_serialize_time_sec = _torch_serialize_with_timing(batch_payload)
        batch_payload["worker_serialize_time_sec"] = float(worker_serialize_time_sec)
        final_payload, _ = _torch_serialize_with_timing(batch_payload)
        return final_payload

    def _materialize_synthetic_torch_batch(
        self,
        batch: batchflow_pb2.Batch,
    ) -> bytes:
        input_shape = _synthetic_torch_input_shape(batch)
        num_classes = _synthetic_torch_num_classes(batch)

        sample_ids: list[str] = []
        source_uris: list[str] = []
        sample_indices: list[int] = []

        for sample in batch.samples:
            sample_ids.append(sample.sample_id)
            source_uris.append(sample.source_uri)
            sample_indices.append(_synthetic_sample_index(sample.source_uri))

        build_start = time.perf_counter()

        inputs = torch.stack(
            [
                _make_synthetic_input_tensor(
                    sample_index=sample_index,
                    input_shape=input_shape,
                )
                for sample_index in sample_indices
            ],
            dim=0,
        )

        labels = torch.tensor(
            [sample_index % num_classes for sample_index in sample_indices],
            dtype=torch.long,
        )

        worker_build_time_sec = time.perf_counter() - build_start

        batch_payload = {
            "x": inputs,
            "label": labels,
            "sample_id": sample_ids,
            "source_uri": source_uris,
            "sample_index": torch.tensor(sample_indices, dtype=torch.long),
            "worker_build_time_sec": float(worker_build_time_sec),
            "worker_serialize_time_sec": 0.0,
            "batch_size": len(sample_indices),
            "batch_id": batch.batch_id,
            "dataset_id": batch.dataset_id,
            "batch_index": int(batch.batch_index),
            "epoch": int(batch.epoch),
            "transform_name": batch.transform_name,
            "payload_format": PayloadFormat.TORCH_BATCH.value,
            "input_shape": list(input_shape),
            "num_classes": int(num_classes),
        }

        payload, worker_serialize_time_sec = _torch_serialize_with_timing(
            batch_payload,
        )

        batch_payload["worker_serialize_time_sec"] = float(worker_serialize_time_sec)

        final_payload, _ = _torch_serialize_with_timing(batch_payload)
        return final_payload

    def _fetch_samples(
        self,
        samples,
    ) -> list[tuple[batchflow_pb2.Sample, bytes, float]]:
        if self.s3_fetch_threads <= 1:
            return [self._fetch_one_sample(sample) for sample in samples]

        with ThreadPoolExecutor(max_workers=self.s3_fetch_threads) as executor:
            return list(executor.map(self._fetch_one_sample, samples))

    def _decode_transform_samples(
        self,
        indexed_fetched: list[tuple[int, batchflow_pb2.Sample, bytes, float]],
        *,
        transform: Callable[[Image.Image], torch.Tensor],
    ) -> list[tuple[int, dict[str, Any]]]:
        if self.decode_threads <= 1:
            return [
                self._decode_transform_one_sample(
                    item,
                    transform=transform,
                )
                for item in indexed_fetched
            ]

        with ThreadPoolExecutor(max_workers=self.decode_threads) as executor:
            return list(
                executor.map(
                    lambda item: self._decode_transform_one_sample(
                        item,
                        transform=transform,
                    ),
                    indexed_fetched,
                )
            )


def _torch_serialize_with_timing(
    batch_payload: dict[str, Any],
) -> tuple[bytes, float]:
    serialize_start = time.perf_counter()
    buffer = io.BytesIO()
    torch.save(batch_payload, buffer)
    serialize_time_sec = time.perf_counter() - serialize_start

    return buffer.getvalue(), serialize_time_sec


def _is_s3_image_batch(batch: batchflow_pb2.Batch) -> bool:
    return bool(batch.samples) and all(
        str(sample.source_uri).startswith("s3://")
        for sample in batch.samples
    )


def _is_synthetic_torch_batch(batch: batchflow_pb2.Batch) -> bool:
    return bool(batch.samples) and all(
        str(sample.source_uri).startswith("memory://synthetic-torch")
        for sample in batch.samples
    )


def _synthetic_sample_index(source_uri: str) -> int:
    try:
        return int(source_uri.rsplit("/", 1)[-1].split("-")[-1])
    except ValueError as exc:
        raise ValueError(f"invalid synthetic sample URI: {source_uri}") from exc


def _make_synthetic_input_tensor(
    *,
    sample_index: int,
    input_shape: tuple[int, ...],
) -> torch.Tensor:
    numel = 1

    for dim in input_shape:
        numel *= dim

    base = torch.arange(numel, dtype=torch.float32).reshape(input_shape)
    return base.add(float(sample_index)).div(float(max(1, numel)))


def _synthetic_torch_input_shape(batch: batchflow_pb2.Batch) -> tuple[int, ...]:
    value = _required_batch_metadata_value(batch, "input_shape")

    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        loaded = [part.strip() for part in value.split(",") if part.strip()]

    input_shape = tuple(int(dim) for dim in loaded)

    if not input_shape:
        raise ValueError("synthetic torch input_shape must be non-empty")

    if any(dim <= 0 for dim in input_shape):
        raise ValueError(
            f"synthetic torch input_shape dimensions must be > 0: {input_shape}"
        )

    return input_shape


def _synthetic_torch_num_classes(batch: batchflow_pb2.Batch) -> int:
    value = _required_batch_metadata_value(batch, "num_classes")
    num_classes = int(value)

    if num_classes <= 0:
        raise ValueError(f"synthetic torch num_classes must be > 0, got {num_classes}")

    return num_classes


def _batch_metadata_dict(batch: batchflow_pb2.Batch) -> dict[str, str]:
    if not hasattr(batch, "metadata"):
        return {}
    return {item.key: item.value for item in batch.metadata}


def _required_batch_metadata_value(
    batch: batchflow_pb2.Batch,
    key: str,
) -> str:
    if not hasattr(batch, "metadata"):
        raise ValueError(
            f"batch proto does not expose metadata; required key={key!r}"
        )

    for item in batch.metadata:
        if item.key == key:
            if item.value == "":
                raise ValueError(f"batch metadata key={key!r} is empty")
            return item.value

    available_keys = [
        item.key
        for item in batch.metadata
    ]

    raise ValueError(
        f"missing required batch metadata key={key!r}; "
        f"available_keys={available_keys}"
    )