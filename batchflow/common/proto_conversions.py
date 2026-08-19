from __future__ import annotations

from typing import Any

from batchflow.common.core import (
    Batch,
    BatchHandle,
    BatchHandleStatus,
    Dataset,
    JobStatus,
    MaterializeBatchTask,
    Sample,
    StorageClass,
)
from batchflow.common.payload_formats import PayloadFormat
from batchflow.proto import batchflow_pb2


def kv_list_to_dict(items) -> dict[str, str]:
    return {item.key: item.value for item in items}


def dict_to_kv_list(d: dict[str, Any] | None) -> list[batchflow_pb2.KeyValue]:
    if not d:
        return []

    return [
        batchflow_pb2.KeyValue(key=str(key), value=str(value))
        for key, value in d.items()
    ]


def storage_class_from_proto(value: int) -> StorageClass:
    if value == batchflow_pb2.STORAGE_CLASS_REUSABLE:
        return StorageClass.REUSABLE

    if value == batchflow_pb2.STORAGE_CLASS_TRANSIENT:
        return StorageClass.TRANSIENT

    raise ValueError(f"unknown storage class value: {value}")


def storage_class_to_proto(value: StorageClass) -> int:
    if value == StorageClass.REUSABLE:
        return batchflow_pb2.STORAGE_CLASS_REUSABLE

    if value == StorageClass.TRANSIENT:
        return batchflow_pb2.STORAGE_CLASS_TRANSIENT

    return batchflow_pb2.STORAGE_CLASS_UNSPECIFIED


def batch_handle_status_from_proto(value: int) -> BatchHandleStatus:
    if value == batchflow_pb2.BATCH_HANDLE_STATUS_PENDING:
        return BatchHandleStatus.PENDING

    if value == batchflow_pb2.BATCH_HANDLE_STATUS_READY:
        return BatchHandleStatus.READY

    if value == batchflow_pb2.BATCH_HANDLE_STATUS_FAILED:
        return BatchHandleStatus.FAILED

    raise ValueError(f"unknown batch handle status value: {value}")


def batch_handle_status_to_proto(value: BatchHandleStatus) -> int:
    if value == BatchHandleStatus.PENDING:
        return batchflow_pb2.BATCH_HANDLE_STATUS_PENDING

    if value == BatchHandleStatus.READY:
        return batchflow_pb2.BATCH_HANDLE_STATUS_READY

    if value == BatchHandleStatus.FAILED:
        return batchflow_pb2.BATCH_HANDLE_STATUS_FAILED

    return batchflow_pb2.BATCH_HANDLE_STATUS_UNSPECIFIED


def job_status_from_proto(value: int) -> JobStatus:
    if value == batchflow_pb2.JOB_STATUS_ACTIVE:
        return JobStatus.ACTIVE

    if value == batchflow_pb2.JOB_STATUS_COMPLETED:
        return JobStatus.COMPLETED

    if value == batchflow_pb2.JOB_STATUS_CANCELLED:
        return JobStatus.CANCELLED

    if value == batchflow_pb2.JOB_STATUS_FAILED:
        return JobStatus.FAILED

    return JobStatus.CANCELLED


def job_status_to_proto(value: JobStatus) -> int:
    if value == JobStatus.ACTIVE:
        return batchflow_pb2.JOB_STATUS_ACTIVE

    if value == JobStatus.COMPLETED:
        return batchflow_pb2.JOB_STATUS_COMPLETED

    if value == JobStatus.CANCELLED:
        return batchflow_pb2.JOB_STATUS_CANCELLED

    if value == JobStatus.FAILED:
        return batchflow_pb2.JOB_STATUS_FAILED

    return batchflow_pb2.JOB_STATUS_UNSPECIFIED


def payload_format_from_string(value: str) -> PayloadFormat:
    if not value:
        raise ValueError("payload_format must be non-empty")

    return PayloadFormat(value)


def payload_format_to_string(value: PayloadFormat | str | None) -> str:
    if value is None:
        return ""

    if isinstance(value, PayloadFormat):
        return value.value

    return PayloadFormat(str(value)).value


def sample_from_proto(pb: batchflow_pb2.Sample) -> Sample:
    return Sample(
        sample_id=pb.sample_id,
        source_uri=pb.source_uri,
        label=int(pb.label) if pb.has_label else None,
        class_name=pb.class_name or None,
        transform_name=pb.transform_name or None,
    )


def sample_to_proto(sample: Sample) -> batchflow_pb2.Sample:
    has_label = sample.label is not None

    return batchflow_pb2.Sample(
        sample_id=sample.sample_id,
        source_uri=sample.source_uri,
        label=int(sample.label) if has_label else 0,
        has_label=has_label,
        class_name=sample.class_name or "",
        transform_name=sample.transform_name or "",
    )


def dataset_from_proto(pb: batchflow_pb2.Dataset) -> Dataset:
    metadata = kv_list_to_dict(pb.metadata)

    payload_format_value = pb.payload_format or metadata.get("payload_format", "")
    dataset_format = pb.dataset_format or metadata.get("format", "")

    return Dataset(
        dataset_id=pb.dataset_id,
        samples=[sample_from_proto(sample) for sample in pb.samples],
        batch_size=int(pb.batch_size),
        payload_format=payload_format_from_string(payload_format_value),
        dataset_format=dataset_format,
        drop_last=bool(pb.drop_last),
        shuffle=bool(pb.shuffle),
        seed=int(pb.seed),
        transform_name=pb.transform_name or None,
        metadata=metadata,
    )


def dataset_to_proto(dataset: Dataset) -> batchflow_pb2.Dataset:
    return batchflow_pb2.Dataset(
        dataset_id=dataset.dataset_id,
        samples=[
            sample_to_proto(sample)
            for sample in dataset.samples
        ],
        batch_size=int(dataset.batch_size),
        drop_last=bool(dataset.drop_last),
        shuffle=bool(dataset.shuffle),
        seed=int(dataset.seed),
        transform_name=dataset.transform_name or "",
        metadata=dict_to_kv_list(dataset.metadata),
        payload_format=payload_format_to_string(dataset.payload_format),
        dataset_format=dataset.dataset_format,
    )


def batch_handle_to_proto(handle: BatchHandle) -> batchflow_pb2.BatchHandle:
    return batchflow_pb2.BatchHandle(
        batch_id=handle.batch_id,
        cache_key=handle.cache_key,
        status=batch_handle_status_to_proto(handle.status),
        fetch_host=handle.fetch_host,
        fetch_port=int(handle.fetch_port),
        fetch_key=handle.fetch_key,
        payload_format=payload_format_to_string(handle.payload_format),
        dataset_format=handle.dataset_format,
        location=handle.location,
        expires_at_ms=handle.expires_at_ms or 0,
        metadata=dict_to_kv_list(handle.metadata),
    )


def batch_handle_from_proto(pb: batchflow_pb2.BatchHandle) -> BatchHandle:
    payload_format = (
        payload_format_from_string(pb.payload_format)
        if pb.payload_format
        else None
    )

    return BatchHandle(
        batch_id=pb.batch_id,
        cache_key=pb.cache_key,
        status=batch_handle_status_from_proto(pb.status),
        fetch_host=pb.fetch_host,
        fetch_port=int(pb.fetch_port),
        fetch_key=pb.fetch_key,
        payload_format=payload_format,
        dataset_format=pb.dataset_format,
        location=pb.location,
        expires_at_ms=int(pb.expires_at_ms) if pb.expires_at_ms > 0 else None,
        metadata=kv_list_to_dict(pb.metadata),
    )


def batch_to_proto(batch: Batch) -> batchflow_pb2.Batch:
    return batchflow_pb2.Batch(
        batch_id=batch.batch_id,
        dataset_id=batch.dataset_id,
        epoch=int(batch.epoch),
        batch_index=int(batch.batch_index),
        batch_size=int(batch.batch_size),
        samples=[
            sample_to_proto(sample)
            for sample in batch.samples
        ],
        planned_at_ms=int(batch.planned_at_ms),
        transform_name=batch.transform_name or "",
        metadata=dict_to_kv_list(batch.metadata),
    )

def batch_from_proto(proto: batchflow_pb2.Batch) -> Batch:
    return Batch(
        batch_id=proto.batch_id,
        dataset_id=proto.dataset_id,
        epoch=int(proto.epoch),
        batch_index=int(proto.batch_index),
        batch_size=int(proto.batch_size),
        samples=[
            sample_from_proto(sample)
            for sample in proto.samples
        ],
        planned_at_ms=int(proto.planned_at_ms),
        transform_name=proto.transform_name or None,
        metadata=kv_list_to_dict(proto.metadata),
    )


def materialize_batch_task_to_proto(
    task: MaterializeBatchTask,
) -> batchflow_pb2.MaterializeBatchTask:
    return batchflow_pb2.MaterializeBatchTask(
        task_id=task.task_id,
        batch=batch_to_proto(task.batch),
        job_id=task.job_id,
        priority=float(task.priority),
        storage_class=storage_class_to_proto(task.storage_class),
        created_at_ms=int(task.created_at_ms),
        not_before_ms=int(task.not_before_ms),
        metadata=dict_to_kv_list(task.metadata),
        payload_format=payload_format_to_string(task.payload_format),
        dataset_format=task.dataset_format,
    )

def task_state_to_proto(task_state) -> batchflow_pb2.MaterializeBatchTask:
    return materialize_batch_task_to_proto(task_state.task)