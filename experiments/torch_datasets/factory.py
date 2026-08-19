from __future__ import annotations

from torch.utils.data import Dataset as TorchDataset

from batchflow.config.config_types import DatasetConfig
from experiments.torch_datasets.s3_image_folder import S3ClassificationDataset
from experiments.torch_datasets.s3_retrieval import S3RetrievalDataset


def build_torch_dataset(config: DatasetConfig) -> TorchDataset:
    if config.dataset_format == "coco_retrieval":
        return S3RetrievalDataset(config=config)

    return S3ClassificationDataset(config=config)
