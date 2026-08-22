from __future__ import annotations

from concurrent import futures

import grpc

from batchflow.common.proto_conversions import (
    batch_handle_to_proto,
    dataset_from_proto,
    dict_to_kv_list,
    job_status_from_proto,
    job_status_to_proto,
    kv_list_to_dict,
    storage_class_from_proto,
    task_state_to_proto,
)
from batchflow.coordinator.service import (
    CoordinatorService,
    TrainerRuntimeFeedback,
)
from batchflow.proto import batchflow_pb2
from batchflow.proto import batchflow_pb2_grpc


class CoordinatorServiceServicer(batchflow_pb2_grpc.CoordinatorServiceServicer):
    def __init__(self, coordinator_service: CoordinatorService) -> None:
        self.coordinator_service = coordinator_service

    def RegisterDataset(self, request, context):
        try:
            dataset = dataset_from_proto(request.dataset)
            self.coordinator_service.register_dataset(dataset)

            return batchflow_pb2.RegisterDatasetResponse(
                dataset_id=dataset.dataset_id,
                sample_count=dataset.sample_count,
                batch_size=dataset.batch_size,
            )
        except Exception as exc:
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

    def StartJob(self, request, context):
        try:
            result = self.coordinator_service.start_job(
                job_id=request.job_id,
                dataset_id=request.dataset_id,
                lookahead_batches=int(request.lookahead_batches),
                metadata=kv_list_to_dict(request.metadata),
            )

            return batchflow_pb2.StartJobResponse(
                job_id=result.job_id,
                epoch=result.epoch,
                next_batch_index=result.next_batch_index,
            )
        except Exception as exc:
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

    
    def GetNextBatch(self, request, context):
        try:
            runtime_feedback = None

            if request.HasField("runtime_feedback"):
                feedback = request.runtime_feedback
                runtime_feedback = TrainerRuntimeFeedback(
                    data_bottleneck_percent=float(feedback.data_bottleneck_percent),
                    avg_data_time_sec=float(feedback.avg_data_time_sec),
                    avg_compute_time_sec=float(feedback.avg_compute_time_sec),
                    avg_coordinator_wait_total_time_sec=float(
                        feedback.avg_coordinator_wait_total_time_sec
                    ),
                    avg_coordinator_pending_polls=float(
                        feedback.avg_coordinator_pending_polls
                    ),
                )

            result = self.coordinator_service.get_next_batch(
                job_id=request.job_id,
                runtime_feedback=runtime_feedback,
            )

            metadata = result.metadata or {}
            done = bool(metadata.get("done", False))

            return batchflow_pb2.GetNextBatchResponse(
                batch_handle=batch_handle_to_proto(result.batch_handle),
                epoch=result.epoch,
                batch_index=result.batch_index or 0,
                has_batch_index=result.batch_index is not None,
                done=done,
                metadata=dict_to_kv_list(metadata),
            )
        except Exception as exc:
            context.abort(grpc.StatusCode.INTERNAL, str(exc))
            

    # Legacy protobuf RPC name; the Python service API calls this acknowledge_batch.
    def CommitBatch(self, request, context):
        try:
            job = self.coordinator_service.acknowledge_batch(
                job_id=request.job_id,
                batch_id=request.batch_id,
                epoch=int(request.epoch),
                batch_index=int(request.batch_index),
            )

            return batchflow_pb2.CommitBatchResponse(
                job_id=job.job_id,
                epoch=job.progress.epoch,
                next_batch_index=job.progress.next_batch_index,
                last_batch_index=job.progress.last_batch_index,
                last_batch_id=job.progress.last_batch_id,
            )
        except Exception as exc:
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

    def FinishJob(self, request, context):
        try:
            result = self.coordinator_service.finish_job(
                job_id=request.job_id,
                reason=request.reason,
                status=job_status_from_proto(request.status),
            )

            return batchflow_pb2.FinishJobResponse(
                job_id=result.job_id,
                status=job_status_to_proto(result.status),
                reason=result.reason,
            )
        except Exception as exc:
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

    def RegisterWorker(self, request, context):
        try:
            worker = self.coordinator_service.register_worker(
                worker_id=request.worker_id,
                hostname=request.hostname,
                metadata=kv_list_to_dict(request.metadata),
            )

            return batchflow_pb2.RegisterWorkerResponse(
                worker_id=worker.worker_id,
            )
        except Exception as exc:
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

    def HeartbeatWorker(self, request, context):
        try:
            worker = self.coordinator_service.heartbeat_worker(request.worker_id)

            return batchflow_pb2.HeartbeatWorkerResponse(
                worker_id=worker.worker_id,
                last_heartbeat_at_ms=worker.last_heartbeat_at_ms,
            )
        except Exception as exc:
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

    def PollMaterializeBatchTask(self, request, context):
        try:
            task_state = self.coordinator_service.poll_materialization_task(
                worker_id=request.worker_id,
            )

            if task_state is None:
                return batchflow_pb2.PollMaterializeBatchTaskResponse(
                    has_task=False,
                )

            return batchflow_pb2.PollMaterializeBatchTaskResponse(
                has_task=True,
                task=task_state_to_proto(task_state),
            )
        except Exception as exc:
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

    def CompleteMaterializeBatchTask(self, request, context):
        try:
            task_state = self.coordinator_service.complete_materialization(
                task_id=request.task_id,
                worker_id=request.worker_id,
                location=request.location,
                size_bytes=int(request.size_bytes),
                storage_class=storage_class_from_proto(request.storage_class),
                expires_at_ms=(
                    int(request.expires_at_ms)
                    if request.expires_at_ms > 0
                    else None
                ),
                metadata=kv_list_to_dict(request.metadata),
            )

            return batchflow_pb2.CompleteMaterializeBatchTaskResponse(
                task_id=task_state.task.task_id,
                batch_id=task_state.task.batch.batch_id,
            )
        except Exception as exc:
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

    def FailMaterializeBatchTask(self, request, context):
        try:
            task_state = self.coordinator_service.fail_materialization(
                task_id=request.task_id,
                reason=request.reason,
            )

            return batchflow_pb2.FailMaterializeBatchTaskResponse(
                task_id=task_state.task.task_id,
                reason=task_state.failure_reason,
            )
        except Exception as exc:
            context.abort(grpc.StatusCode.INTERNAL, str(exc))


def serve_grpc(
    coordinator_service: CoordinatorService,
    host: str = "127.0.0.1",
    port: int = 50051,
    grpc_max_workers: int = 8,
) -> grpc.Server:
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=grpc_max_workers),
        options=[
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
        ],
    )

    batchflow_pb2_grpc.add_CoordinatorServiceServicer_to_server(
        CoordinatorServiceServicer(coordinator_service),
        server,
    )

    server.add_insecure_port(f"{host}:{port}")
    server.start()

    return server