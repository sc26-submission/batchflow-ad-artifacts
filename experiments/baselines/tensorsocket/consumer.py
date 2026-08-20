from __future__ import annotations

import logging
import threading
import time
import uuid
from queue import Empty, Queue
from typing import Any, Iterator

import zmq

from experiments.baselines.tensorsocket.payload import unpack_tensors


LOGGER = logging.getLogger("experiments.baselines.tensorsocket.consumer")


class TensorConsumer(Iterator[dict[str, Any]]):
    """Receive shared tensor batches from a TensorSocket producer."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 5555,
        control_port: int = 5556,
        buffer_size: int = 8,
        receive_timeout_seconds: float = 120.0,
    ) -> None:
        if buffer_size <= 0:
            raise ValueError("buffer_size must be > 0")

        self.consumer_id = uuid.uuid4().hex
        self.receive_timeout_seconds = receive_timeout_seconds
        self.buffer: Queue[dict[str, Any]] = Queue(maxsize=buffer_size)
        self._stop_event = threading.Event()
        self._closed = False
        self._last_received = -1

        self.context = zmq.Context()

        self.data_socket = self.context.socket(zmq.SUB)
        self.data_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.data_socket.setsockopt(zmq.RCVHWM, max(100, buffer_size * 4))
        self.data_socket.connect(f"tcp://{host}:{port}")

        self.control_socket = self.context.socket(zmq.PUSH)
        self.control_socket.setsockopt(zmq.SNDHWM, 1000)
        self.control_socket.connect(f"tcp://{host}:{control_port}")

        self._send_control("register")

        self.fetch_thread = threading.Thread(
            target=self._fetch_loop,
            name=f"tensorsocket-fetch-{self.consumer_id[:8]}",
            daemon=True,
        )
        self.fetch_thread.start()

    def __iter__(self) -> TensorConsumer:
        return self

    def __next__(self) -> dict[str, Any]:
        wait_start = time.perf_counter()
        cache_hit = not self.buffer.empty()
        try:
            message = self.buffer.get(timeout=self.receive_timeout_seconds)
        except Empty as exc:
            raise RuntimeError(
                "timed out waiting for a TensorSocket batch from the producer"
            ) from exc

        wait_time = time.perf_counter() - wait_start

        if message.get("type") == "shutdown":
            raise StopIteration

        tensor_keys = tuple(message.get("tensor_keys", ("image", "label")))
        tensors = unpack_tensors(message["data"])
        if len(tensor_keys) != len(tensors):
            raise RuntimeError(
                f"TensorSocket payload has {len(tensors)} tensors but {len(tensor_keys)} keys"
            )

        metrics = dict(message.get("metrics", {}))
        batch = dict(zip(tensor_keys, tensors))
        batch.update({
            "batch_indices": metrics.get("batch_indices"),
            "io_time_sec": metrics.get("io_time_sec", 0.0),
            "decode_time_sec": metrics.get("decode_time_sec", 0.0),
            "transform_time_sec": metrics.get("transform_time_sec", 0.0),
            "tensorsocket_wait_time_sec": wait_time,
            "tensorsocket_cache_hit": int(cache_hit),
            # "batch_index": int(message.get("batch_index", -1)),
            # "epoch": int(message.get("epoch", 0)),
        })
        return batch

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        self._stop_event.set()

        try:
            self._send_control("close")
        except zmq.ZMQError:
            pass

        self.fetch_thread.join(timeout=1.0)
        self.data_socket.close(linger=0)
        self.control_socket.close(linger=0)
        self.context.term()

    def _fetch_loop(self) -> None:
        while not self._stop_event.is_set():
            if not self.data_socket.poll(100, zmq.POLLIN):
                continue

            try:
                payload = self.data_socket.recv_pyobj()
            except zmq.ZMQError:
                return

            if payload.get("type") == "shutdown":
                self.buffer.put({"type": "shutdown"})
                return

            if payload.get("type") != "data":
                continue

            messages = payload.get("consumers", {}).get(self.consumer_id, [])

            for message in messages:
                batch_index = int(message.get("batch_index", -1))

                if batch_index <= self._last_received:
                    continue

                self.buffer.put(message)
                self._last_received = batch_index
                self._send_control("ack", batch_index=batch_index)

    def _send_control(self, message_type: str, **values: Any) -> None:
        self.control_socket.send_pyobj(
            {
                "type": message_type,
                "consumer_id": self.consumer_id,
                **values,
            }
        )
