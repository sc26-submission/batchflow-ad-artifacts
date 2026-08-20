from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterator

import torch
import zmq

from experiments.baselines.tensorsocket.payload import pack_tensors


LOGGER = logging.getLogger("experiments.baselines.tensorsocket.producer")


@dataclass
class ConsumerState:
    last_sent: int = -1
    last_acked: int = -1

    @property
    def outstanding(self) -> int:
        return self.last_sent - self.last_acked


@dataclass
class SharedBatch:
    batch_index: int
    epoch: int
    tensor_keys: tuple[str, ...]
    tensors: tuple[torch.Tensor, ...]
    metrics: dict[str, float]


class TensorProducer:
    """Share prepared PyTorch batches with multiple trainer processes.

    The producer reads and transforms each source batch once, then publishes the
    resulting tensors to all consumers over ZeroMQ. CPU tensor storage is shared
    rather than copied into each trainer process.
    """

    def __init__(
        self,
        data_loader: Any,
        *,
        expected_consumers: int,
        port: int = 5555,
        control_port: int = 5556,
        consumer_buffer_size: int = 8,
        max_lag_batches: int = 64,
        bind_host: str = "*",
    ) -> None:
        if expected_consumers <= 0:
            raise ValueError("expected_consumers must be > 0")
        if consumer_buffer_size <= 0:
            raise ValueError("consumer_buffer_size must be > 0")
        if max_lag_batches < consumer_buffer_size:
            raise ValueError("max_lag_batches must be >= consumer_buffer_size")

        self.data_loader = data_loader
        self.loader_iter: Iterator = iter(data_loader)
        self.expected_consumers = expected_consumers
        self.consumer_buffer_size = consumer_buffer_size
        self.max_lag_batches = max_lag_batches

        self.epoch = 0
        self.next_batch_index = 0
        self.consumers: dict[str, ConsumerState] = {}
        self.buffer: deque[SharedBatch] = deque()
        self.buffer_by_index: dict[int, SharedBatch] = {}

        self.context = zmq.Context()
        self.data_socket = self.context.socket(zmq.PUB)
        self.data_socket.setsockopt(zmq.SNDHWM, max(100, consumer_buffer_size * expected_consumers * 4))
        self.data_socket.bind(f"tcp://{bind_host}:{port}")

        self.control_socket = self.context.socket(zmq.PULL)
        self.control_socket.setsockopt(zmq.RCVHWM, 1000)
        self.control_socket.bind(f"tcp://{bind_host}:{control_port}")

        self._closed = False

    def run(self, stop_event: Any) -> None:
        self._wait_for_consumers(stop_event)

        if stop_event.is_set():
            return

        # Avoid the PUB/SUB slow-joiner race after all registrations arrive.
        time.sleep(0.25)
        LOGGER.info("TensorSocket producer ready consumers=%d", len(self.consumers))

        while not stop_event.is_set():
            self._drain_control()

            if not self.consumers:
                time.sleep(0.01)
                continue

            self._prune_buffer()
            messages = self._build_messages()

            if messages:
                self.data_socket.send_pyobj({"type": "data", "consumers": messages})
            else:
                time.sleep(0.001)

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True

        try:
            self.data_socket.send_pyobj({"type": "shutdown"}, flags=zmq.NOBLOCK)
        except zmq.ZMQError:
            pass

        self.data_socket.close(linger=0)
        self.control_socket.close(linger=0)
        self.context.term()

    def _wait_for_consumers(self, stop_event: Any) -> None:
        LOGGER.info(
            "Waiting for TensorSocket consumers expected=%d",
            self.expected_consumers,
        )

        while len(self.consumers) < self.expected_consumers:
            if stop_event.is_set():
                return

            self._drain_control(block_ms=100)

        LOGGER.info("All TensorSocket consumers registered")

    def _drain_control(self, block_ms: int = 0) -> None:
        first = True

        while True:
            timeout = block_ms if first else 0
            first = False

            if not self.control_socket.poll(timeout, zmq.POLLIN):
                return

            message = self.control_socket.recv_pyobj()
            message_type = message.get("type")
            consumer_id = str(message.get("consumer_id", ""))

            if not consumer_id:
                continue

            if message_type == "register":
                if consumer_id not in self.consumers:
                    self.consumers[consumer_id] = ConsumerState()
                    LOGGER.info("Consumer registered id=%s", consumer_id)
                continue

            if message_type == "ack":
                state = self.consumers.get(consumer_id)
                if state is not None:
                    state.last_acked = max(
                        state.last_acked,
                        int(message.get("batch_index", -1)),
                    )
                continue

            if message_type == "close":
                if consumer_id in self.consumers:
                    self.consumers.pop(consumer_id, None)
                    LOGGER.info("Consumer closed id=%s", consumer_id)

    def _build_messages(self) -> dict[str, list[dict[str, Any]]]:
        if not self.consumers:
            return {}

        slowest_acked = min(state.last_acked for state in self.consumers.values())
        max_allowed_index = slowest_acked + self.max_lag_batches

        requested: dict[str, list[int]] = {}
        max_needed = -1

        for consumer_id, state in self.consumers.items():
            capacity = self.consumer_buffer_size - state.outstanding
            if capacity <= 0:
                continue

            start = state.last_sent + 1
            end = min(start + capacity - 1, max_allowed_index)

            if end < start:
                continue

            indices = list(range(start, end + 1))
            requested[consumer_id] = indices
            max_needed = max(max_needed, end)

        if max_needed < 0:
            return {}

        self._fill_buffer_through(max_needed)

        packed: dict[int, tuple] = {}
        messages: dict[str, list[dict[str, Any]]] = {}

        for consumer_id, indices in requested.items():
            consumer_messages: list[dict[str, Any]] = []

            for batch_index in indices:
                batch = self.buffer_by_index.get(batch_index)
                if batch is None:
                    break

                if batch_index not in packed:
                    packed[batch_index] = pack_tensors(batch.tensors)

                consumer_messages.append(
                    {
                        "batch_index": batch.batch_index,
                        "epoch": batch.epoch,
                        "tensor_keys": batch.tensor_keys,
                        "data": packed[batch_index],
                        "metrics": batch.metrics,
                    }
                )

            if consumer_messages:
                messages[consumer_id] = consumer_messages
                self.consumers[consumer_id].last_sent = consumer_messages[-1]["batch_index"]

        return messages

    def _fill_buffer_through(self, batch_index: int) -> None:
        while self.next_batch_index <= batch_index:
            source_batch = self._next_source_batch()
            shared_batch = self._prepare_source_batch(source_batch)

            self.buffer.append(shared_batch)
            self.buffer_by_index[shared_batch.batch_index] = shared_batch
            self.next_batch_index += 1

    def _next_source_batch(self) -> dict[str, Any]:
        try:
            return next(self.loader_iter)
        except StopIteration:
            self.epoch += 1
            self.loader_iter = iter(self.data_loader)
            return next(self.loader_iter)

    def _prepare_source_batch(self, batch: dict[str, Any]) -> SharedBatch:
        if all(key in batch for key in ("image", "text", "text_atts", "idx")):
            tensor_keys = ("image", "text", "text_atts", "idx")
        elif all(key in batch for key in ("image", "label")):
            tensor_keys = ("image", "label")
        else:
            raise TypeError(
                "TensorSocket producer expects classification or retrieval tensor batches"
            )

        tensors = tuple(batch[key] for key in tensor_keys)
        if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise TypeError("TensorSocket producer can only share tensor-valued batch fields")

        metrics = {
            "batch_indices": batch.get("batch_indices", batch.get("index").tolist()),
            "io_time_sec": self._sum_metric(batch.get("io_time_sec")),
            "decode_time_sec": self._sum_metric(batch.get("decode_time_sec")),
            "transform_time_sec": self._sum_metric(batch.get("transform_time_sec")),
        }

        return SharedBatch(
            batch_index=self.next_batch_index,
            epoch=self.epoch,
            tensor_keys=tensor_keys,
            tensors=tuple(tensor.contiguous() for tensor in tensors),
            metrics=metrics,
        )

    @staticmethod
    def _sum_metric(value: Any) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.sum().item())

        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _prune_buffer(self) -> None:
        if not self.consumers:
            self.buffer.clear()
            self.buffer_by_index.clear()
            return

        min_acked = min(state.last_acked for state in self.consumers.values())

        while self.buffer and self.buffer[0].batch_index <= min_acked:
            batch = self.buffer.popleft()
            self.buffer_by_index.pop(batch.batch_index, None)
