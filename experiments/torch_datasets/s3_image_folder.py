from __future__ import annotations

import time
from dataclasses import dataclass
from io import BytesIO
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset as TorchDataset

from batchflow.common.core import Dataset as BatchFlowDataset
from batchflow.common.s3io import read_s3_bytes
from batchflow.common.transforms import build_image_transform
from batchflow.config.config_types import DatasetConfig
from batchflow.coordinator.dataset_builder import build_dataset_from_config


@dataclass(frozen=True)
class S3ImageSample:
    uri: str
    class_name: str
    label: int
    split: str


def samples_from_batchflow_dataset(
    dataset: BatchFlowDataset,
    *,
    split: str,
) -> tuple[list[S3ImageSample], dict[str, int]]:
    samples: list[S3ImageSample] = []
    class_to_idx: dict[str, int] = {}

    for sample in dataset.samples:
        if sample.class_name is None:
            raise ValueError(f"sample {sample.sample_id!r} has no class_name")
        if sample.label is None:
            raise ValueError(f"sample {sample.sample_id!r} has no label")

        class_name = sample.class_name
        label = int(sample.label)
        existing_label = class_to_idx.get(class_name)

        if existing_label is not None and existing_label != label:
            raise ValueError(
                f"inconsistent label for class {class_name!r}: "
                f"found both {existing_label} and {label}"
            )

        class_to_idx[class_name] = label
        samples.append(
            S3ImageSample(
                uri=sample.source_uri,
                class_name=class_name,
                label=label,
                split=split,
            )
        )

    if not samples:
        raise ValueError(f"dataset {dataset.dataset_id!r} contains no samples")

    class_to_idx = dict(sorted(class_to_idx.items(), key=lambda item: item[1]))
    return samples, class_to_idx


class S3ClassificationDataset(TorchDataset):
    """Torch dataset backed by the same BatchFlow dataset index.

    This supports both class-folder datasets such as ImageNet and flat
    annotation-backed datasets such as Open Images. The system runners do not
    need dataset-specific implementations.
    """

    def __init__(
        self,
        *,
        config: DatasetConfig,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        force_rebuild_manifest: bool = False,
    ) -> None:
        if not config.prefix_uri.startswith("s3://"):
            raise ValueError(
                f"S3ClassificationDataset requires an S3 URI, got {config.prefix_uri!r}"
            )

        self.config = config
        dataset = build_dataset_from_config(
            config,
            force_rebuild=force_rebuild_manifest,
        )
        self.samples, self.class_to_idx = samples_from_batchflow_dataset(
            dataset,
            split=config.split,
        )
        self.transform = transform or build_image_transform(
            config.transform_name or "resize_224"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]

        io_start = time.perf_counter()
        raw = read_s3_bytes(sample.uri)
        io_time_sec = time.perf_counter() - io_start

        decode_start = time.perf_counter()
        with Image.open(BytesIO(raw)) as image:
            image = image.convert("RGB")
            decode_time_sec = time.perf_counter() - decode_start

            transform_start = time.perf_counter()
            image_tensor = self.transform(image)
            transform_time_sec = time.perf_counter() - transform_start

        return {
            "image": image_tensor,
            "label": sample.label,
            "index": index,
            "source_uri": sample.uri,
            "class_name": sample.class_name,
            "io_time_sec": torch.tensor(io_time_sec, dtype=torch.float32),
            "decode_time_sec": torch.tensor(decode_time_sec, dtype=torch.float32),
            "transform_time_sec": torch.tensor(transform_time_sec, dtype=torch.float32),
        }
