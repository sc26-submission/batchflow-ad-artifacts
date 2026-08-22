from __future__ import annotations

from typing import Any, Optional

import grpc

from batchflow.proto import batchflow_pb2
from batchflow.proto import batchflow_pb2_grpc


def _dict_to_kv_list(
    metadata: dict[str, Any] | None,
) -> list[batchflow_pb2.KeyValue]:
    return [
        batchflow_pb2.KeyValue(key=str(key), value=str(value))
        for key, value in (metadata or {}).items()
    ]


class CoordinatorGrpcClient:
    def __init__(self, coordinator_address: str) -> None:
        self.coordinator_address = coordinator_address
        self.channel: Optional[grpc.Channel] = None
        self.stub: Optional[batchflow_pb2_grpc.CoordinatorServiceStub] = None

    def connect(self) -> None:
        self.channel = grpc.insecure_channel(
            self.coordinator_address,
            options=[
                ("grpc.max_send_message_length", 256 * 1024 * 1024),
                ("grpc.max_receive_message_length", 256 * 1024 * 1024),
            ],
        )
        self.stub = batchflow_pb2_grpc.CoordinatorServiceStub(self.channel)

    def close(self) -> None:
        if self.channel is not None:
            self.channel.close()
            self.channel = None
            self.stub = None

    def _require_stub(self) -> batchflow_pb2_grpc.CoordinatorServiceStub:
        if self.stub is None:
            raise RuntimeError(
                f"CoordinatorGrpcClient is not connected: "
                f"coordinator_address={self.coordinator_address}"
            )

        return self.stub

    def register_worker(
        self,
        worker_id: str,
        hostname: str,
        metadata: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> batchflow_pb2.RegisterWorkerResponse:
        stub = self._require_stub()

        return stub.RegisterWorker(
            batchflow_pb2.RegisterWorkerRequest(
                worker_id=worker_id,
                hostname=hostname,
                metadata=_dict_to_kv_list(metadata),
            ),
            timeout=timeout_seconds,
        )

    def heartbeat_worker(
        self,
        worker_id: str,
        timeout_seconds: float | None = None,
    ) -> batchflow_pb2.HeartbeatWorkerResponse:
        stub = self._require_stub()

        return stub.HeartbeatWorker(
            batchflow_pb2.HeartbeatWorkerRequest(
                worker_id=worker_id,
            ),
            timeout=timeout_seconds,
        )

    def poll_materialization_task(
        self,
        worker_id: str,
        timeout_seconds: float | None = None,
    ) -> batchflow_pb2.PollMaterializeBatchTaskResponse:
        stub = self._require_stub()

        return stub.PollMaterializeBatchTask(
            batchflow_pb2.PollMaterializeBatchTaskRequest(
                worker_id=worker_id,
            ),
            timeout=timeout_seconds,
        )

    def complete_materialization(
        self,
        *,
        task_id: str,
        worker_id: str,
        location: str,
        size_bytes: int,
        storage_class: int,
        expires_at_ms: int = 0,
        metadata: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> batchflow_pb2.CompleteMaterializeBatchTaskResponse:
        stub = self._require_stub()

        return stub.CompleteMaterializeBatchTask(
            batchflow_pb2.CompleteMaterializeBatchTaskRequest(
                task_id=task_id,
                worker_id=worker_id,
                location=location,
                size_bytes=int(size_bytes),
                storage_class=storage_class,
                expires_at_ms=int(expires_at_ms or 0),
                metadata=_dict_to_kv_list(metadata),
            ),
            timeout=timeout_seconds,
        )

    def fail_materialization(
        self,
        *,
        task_id: str,
        worker_id: str,
        reason: str,
        timeout_seconds: float | None = None,
    ) -> batchflow_pb2.FailMaterializeBatchTaskResponse:
        stub = self._require_stub()

        return stub.FailMaterializeBatchTask(
            batchflow_pb2.FailMaterializeBatchTaskRequest(
                task_id=task_id,
                worker_id=worker_id,
                reason=reason,
            ),
            timeout=timeout_seconds,
        )

    def start_job(
        self,
        *,
        job_id: str,
        dataset_id: str,
        lookahead_batches: int,
        metadata: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> batchflow_pb2.StartJobResponse:
        stub = self._require_stub()

        return stub.StartJob(
            batchflow_pb2.StartJobRequest(
                job_id=job_id,
                dataset_id=dataset_id,
                lookahead_batches=int(lookahead_batches),
                metadata=_dict_to_kv_list(metadata),
            ),
            timeout=timeout_seconds,
        )

    def get_next_batch(
        self,
        job_id: str,
        *,
        runtime_feedback: Any | None = None,
        timeout_seconds: float | None = None,
    ) -> batchflow_pb2.GetNextBatchResponse:
        stub = self._require_stub()

        request = batchflow_pb2.GetNextBatchRequest(
            job_id=job_id,
        )

        if runtime_feedback is not None:
            request.runtime_feedback.CopyFrom(
                batchflow_pb2.TrainerRuntimeFeedback(
                    data_bottleneck_percent=float(
                        runtime_feedback.data_bottleneck_percent
                    ),
                    avg_data_time_sec=float(runtime_feedback.avg_data_time_sec),
                    avg_compute_time_sec=float(runtime_feedback.avg_compute_time_sec),
                    avg_coordinator_wait_total_time_sec=float(
                        runtime_feedback.avg_coordinator_wait_total_time_sec
                    ),
                    avg_coordinator_pending_polls=float(
                        runtime_feedback.avg_coordinator_pending_polls
                    ),
                )
            )

        return stub.GetNextBatch(
            request,
            timeout=timeout_seconds,
        )

    def acknowledge_batch(
        self,
        *,
        job_id: str,
        batch_id: str,
        epoch: int,
        batch_index: int,
        timeout_seconds: float | None = None,
    ) -> batchflow_pb2.CommitBatchResponse:
        stub = self._require_stub()

        # The protobuf method keeps its original wire name for compatibility.
        return stub.CommitBatch(
            batchflow_pb2.CommitBatchRequest(
                job_id=job_id,
                batch_id=batch_id,
                epoch=int(epoch),
                batch_index=int(batch_index),
            ),
            timeout=timeout_seconds,
        )

    def finish_job(
        self,
        *,
        job_id: str,
        reason: str,
        status: int,
        timeout_seconds: float | None = None,
    ) -> batchflow_pb2.FinishJobResponse:
        stub = self._require_stub()

        return stub.FinishJob(
            batchflow_pb2.FinishJobRequest(
                job_id=job_id,
                reason=reason,
                status=status,
            ),
            timeout=timeout_seconds,
        )
