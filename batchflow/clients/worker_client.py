from __future__ import annotations

from typing import Optional

import grpc

from batchflow.proto import batchflow_pb2
from batchflow.proto import batchflow_pb2_grpc


class WorkerFetchClient:
    def __init__(self) -> None:
        self._channels: dict[str, grpc.Channel] = {}
        self._stubs: dict[str, batchflow_pb2_grpc.WorkerDataServiceStub] = {}

    def fetch_batch(
        self,
        *,
        host: str,
        port: int,
        key: str,
        timeout_seconds: Optional[float] = None,
    ) -> bytes:
        addr = f"{host}:{port}"
        stub = self._get_stub(addr)

        resp = stub.FetchBatch(
            batchflow_pb2.FetchBatchRequest(
                key=key,
            ),
            timeout=timeout_seconds,
        )

        return resp.payload

    def close(self) -> None:
        for channel in self._channels.values():
            channel.close()

        self._channels.clear()
        self._stubs.clear()

    def _get_stub(self, addr: str) -> batchflow_pb2_grpc.WorkerDataServiceStub:
        if addr not in self._stubs:
            channel = grpc.insecure_channel(
                addr,
                options=[
                    ("grpc.max_send_message_length", 256 * 1024 * 1024),
                    ("grpc.max_receive_message_length", 256 * 1024 * 1024),
                ],
            )
            self._channels[addr] = channel
            self._stubs[addr] = batchflow_pb2_grpc.WorkerDataServiceStub(channel)

        return self._stubs[addr]