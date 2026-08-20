from __future__ import annotations

import logging
from concurrent import futures

import grpc

from batchflow.data_worker.payload_store import InMemoryPayloadStore
from batchflow.proto import batchflow_pb2
from batchflow.proto import batchflow_pb2_grpc


LOGGER = logging.getLogger("batchflow.worker")


class WorkerDataService(batchflow_pb2_grpc.WorkerDataServiceServicer):
    def __init__(self, payload_store: InMemoryPayloadStore) -> None:
        self.payload_store = payload_store

    def FetchBatch(self, request, context):
        entry = self.payload_store.get(key=request.key)

        if entry is None:
            LOGGER.warning(
                "Batch fetch missed | key=%s | local_payloads=%s",
                request.key,
                self.payload_store.size(),
            )
            context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"batch payload not found key={request.key}",
            )
            raise RuntimeError("unreachable after context.abort")

        LOGGER.debug(
            "Batch served | key=%s | size=%s | format=%s",
            request.key,
            entry.size_bytes,
            entry.payload_format,
        )

        return batchflow_pb2.FetchBatchResponse(
            key=request.key,
            payload=entry.payload,
        )


def serve_worker_data_grpc(
    *,
    payload_store: InMemoryPayloadStore,
    host: str,
    port: int,
    grpc_thread_pool_size: int = 8,
) -> grpc.Server:
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=grpc_thread_pool_size),
        options=[
            ("grpc.max_send_message_length", 256 * 1024 * 1024),
            ("grpc.max_receive_message_length", 256 * 1024 * 1024),
        ],
    )

    batchflow_pb2_grpc.add_WorkerDataServiceServicer_to_server(
        WorkerDataService(payload_store),
        server,
    )

    server.add_insecure_port(f"{host}:{port}")
    server.start()

    return server