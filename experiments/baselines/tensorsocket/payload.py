from __future__ import annotations

from typing import Any
import warnings

from torch import Tensor
from torch.multiprocessing.reductions import rebuild_cuda_tensor, rebuild_tensor


class TensorPayload:
    """Pickle-friendly wrapper for shared PyTorch tensor storage.

    CPU tensors are moved into shared memory before transmission. CUDA tensor
    support is retained for compatibility with the original TensorSocket code,
    although the experiment runner keeps producer batches on CPU so each trainer
    can transfer its batch to its own GPU.
    """

    def __init__(self, tensor: Tensor | dict[str, Any]) -> None:
        if isinstance(tensor, Tensor):
            self.payload = self._from_tensor(tensor)
            self._tensor = tensor
            return

        self.payload = tensor

        if "storage_cls" in tensor:
            self._tensor = rebuild_cuda_tensor(Tensor, **tensor)
        else:
            self._tensor = rebuild_tensor(
                tensor["cls"],
                tensor["storage"],
                tensor["metadata"],
            )

    @staticmethod
    def _from_tensor(tensor: Tensor) -> dict[str, Any]:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="TypedStorage is deprecated.*",
                category=UserWarning,
            )
            storage = tensor._typed_storage()

        if tensor.is_cuda:
            (
                storage_device,
                storage_handle,
                storage_size_bytes,
                storage_offset_bytes,
                ref_counter_handle,
                ref_counter_offset,
                event_handle,
                event_sync_required,
            ) = storage._share_cuda_()

            return {
                "dtype": tensor.dtype,
                "tensor_size": tuple(tensor.size()),
                "tensor_stride": tensor.stride(),
                "tensor_offset": tensor.storage_offset(),
                "storage_cls": type(storage),
                "storage_device": storage_device,
                "storage_handle": storage_handle,
                "storage_size_bytes": int(storage_size_bytes),
                "storage_offset_bytes": storage_offset_bytes,
                "requires_grad": False,
                "ref_counter_handle": ref_counter_handle,
                "ref_counter_offset": ref_counter_offset,
                "event_handle": event_handle,
                "event_sync_required": event_sync_required,
            }

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="TypedStorage is deprecated.*",
                category=UserWarning,
            )
            storage.share_memory_()

        metadata = (
            tensor.storage_offset(),
            tensor.size(),
            tensor.stride(),
            tensor.requires_grad,
        )

        return {
            "storage": storage,
            "cls": type(storage),
            "metadata": metadata,
        }

    def __reduce__(self) -> tuple[type[TensorPayload], tuple[dict[str, Any]]]:
        return self.__class__, (self.payload,)

    @property
    def tensor(self) -> Tensor:
        return self._tensor


def pack_tensors(data: tuple[Tensor, ...]) -> tuple[TensorPayload, ...]:
    return tuple(TensorPayload(tensor) for tensor in data)


def unpack_tensors(data: tuple[TensorPayload, ...]) -> tuple[Tensor, ...]:
    return tuple(item.tensor for item in data)
