from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from batchflow.config.config_types import DatasetConfig, merge_dataclass


_CONFIG_ROOT = Path(__file__).resolve().parent
_DATASET_ROOT = _CONFIG_ROOT / "dataset"


def load_dataset_config(name: str) -> DatasetConfig:
    """Load one named dataset config from batchflow/config/dataset/."""

    clean_name = str(name).strip()
    if not clean_name or Path(clean_name).name != clean_name:
        raise ValueError(f"Invalid dataset config name: {name!r}")

    path = _DATASET_ROOT / f"{clean_name}.yaml"
    if not path.is_file():
        available = ", ".join(sorted(p.stem for p in _DATASET_ROOT.glob("*.yaml")))
        raise FileNotFoundError(
            f"Dataset config {clean_name!r} was not found at {path}. "
            f"Available datasets: {available or '<none>'}"
        )

    config = merge_dataclass(DatasetConfig, OmegaConf.load(path))
    config.input_shape = tuple(config.input_shape)
    return config
