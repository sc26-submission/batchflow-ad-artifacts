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
from batchflow.common.transforms import build_image_transform, build_text_transform
from batchflow.config.config_types import DatasetConfig
from batchflow.coordinator.dataset_builder import build_dataset_from_config


@dataclass(frozen=True)
class S3RetrievalSample:
    uri: str
    caption: str
    image_index: int


def samples_from_batchflow_dataset(dataset: BatchFlowDataset) -> list[S3RetrievalSample]:
    samples: list[S3RetrievalSample] = []

    for sample in dataset.samples:
        if sample.class_name is None:
            raise ValueError(f"retrieval sample {sample.sample_id!r} has no caption")
        if sample.label is None:
            raise ValueError(f"retrieval sample {sample.sample_id!r} has no image index")

        samples.append(
            S3RetrievalSample(
                uri=sample.source_uri,
                caption=sample.class_name,
                image_index=int(sample.label),
            )
        )

    if not samples:
        raise ValueError(f"dataset {dataset.dataset_id!r} contains no retrieval samples")

    return samples


class S3RetrievalDataset(TorchDataset):
    """COCO image-text retrieval dataset backed by the BatchFlow index."""

    def __init__(
        self,
        *,
        config: DatasetConfig,
        image_transform: Callable[[Image.Image], torch.Tensor] | None = None,
        text_transform: Callable[[str], torch.Tensor] | None = None,
        force_rebuild_manifest: bool = False,
    ) -> None:
        if not config.prefix_uri.startswith("s3://"):
            raise ValueError(
                f"S3RetrievalDataset requires an S3 URI, got {config.prefix_uri!r}"
            )
        if config.dataset_format != "coco_retrieval":
            raise ValueError(
                f"S3RetrievalDataset requires dataset_format='coco_retrieval', "
                f"got {config.dataset_format!r}"
            )

        self.config = config
        dataset = build_dataset_from_config(
            config,
            force_rebuild=force_rebuild_manifest,
        )
        self.samples = samples_from_batchflow_dataset(dataset)
        self.image_transform = image_transform or build_image_transform(
            config.transform_name or "albef_train_384"
        )
        self.text_transform = text_transform or build_text_transform(
            config.text_transform_name or "albef_text_30"
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
            image_tensor = self.image_transform(image)
            text_tensor = self.text_transform(sample.caption)
            transform_time_sec = time.perf_counter() - transform_start

        text_atts = (text_tensor != 0).long()

        return {
            "image": image_tensor,
            "text": text_tensor,
            "text_atts": text_atts,
            "idx": sample.image_index,
            "source_uri": sample.uri,
            "io_time_sec": torch.tensor(io_time_sec, dtype=torch.float32),
            "decode_time_sec": torch.tensor(decode_time_sec, dtype=torch.float32),
            "transform_time_sec": torch.tensor(transform_time_sec, dtype=torch.float32),
        }
