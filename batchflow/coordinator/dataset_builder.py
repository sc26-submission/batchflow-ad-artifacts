from __future__ import annotations

import csv
import io
import logging
from typing import Any, Iterator

from batchflow.common.core import Dataset, Sample
from batchflow.common.payload_formats import PayloadFormat
from batchflow.common.s3io import (
    get_s3_client,
    iter_s3_prefix,
    parse_s3_uri,
    read_s3_json,
    write_s3_json,
)
from batchflow.config.config_types import DatasetConfig

LOGGER = logging.getLogger("batchflow.service")

IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
)


def build_dataset_from_config(
    dataset_config: DatasetConfig,
    *,
    force_rebuild: bool = False,
) -> Dataset:
    """Build the dataset described by one canonical DatasetConfig."""
    prefix_uri = str(dataset_config.prefix_uri)

    if _is_synthetic_torch_memory_uri(prefix_uri):
        return build_synthetic_torch_dataset(
            dataset_id=str(dataset_config.dataset_id),
            prefix_uri=prefix_uri,
            num_samples=dataset_config.num_samples,
            batch_size=dataset_config.batch_size,
            input_shape=tuple(dataset_config.input_shape),
            num_classes=dataset_config.num_classes,
            transform_name=dataset_config.transform_name,
            drop_last=dataset_config.drop_last,
            shuffle=dataset_config.shuffle,
            seed=dataset_config.seed,
        )

    if not prefix_uri.startswith("s3://"):
        raise ValueError(f"Unsupported dataset prefix_uri: {prefix_uri}")

    dataset_format = str(dataset_config.dataset_format).strip().lower()

    if dataset_format == "image_folder":
        return build_s3_imagefolder_dataset(
            dataset_id=str(dataset_config.dataset_id),
            prefix_uri=prefix_uri,
            batch_size=dataset_config.batch_size,
            split=str(dataset_config.split),
            transform_name=dataset_config.transform_name,
            drop_last=dataset_config.drop_last,
            shuffle=dataset_config.shuffle,
            seed=dataset_config.seed,
            force_rebuild=force_rebuild,
        )

    if dataset_format == "openimages_boxable":
        if not dataset_config.annotations_uri:
            raise ValueError(
                "Open Images requires dataset.annotations_uri pointing to the "
                "human image-level annotation CSV"
            )

        return build_s3_openimages_dataset(
            dataset_id=str(dataset_config.dataset_id),
            prefix_uri=prefix_uri,
            batch_size=dataset_config.batch_size,
            split=str(dataset_config.split),
            annotations_uri=str(dataset_config.annotations_uri),
            class_descriptions_uri=(
                str(dataset_config.class_descriptions_uri)
                if dataset_config.class_descriptions_uri
                else None
            ),
            transform_name=dataset_config.transform_name,
            drop_last=dataset_config.drop_last,
            shuffle=dataset_config.shuffle,
            seed=dataset_config.seed,
            force_rebuild=force_rebuild,
        )

    if dataset_format == "coco_retrieval":
        if not dataset_config.annotations_uri:
            raise ValueError(
                "COCO retrieval requires dataset.annotations_uri pointing to "
                "captions_train2014.json (or an equivalent COCO captions file)"
            )

        return build_s3_coco_retrieval_dataset(
            dataset_id=str(dataset_config.dataset_id),
            prefix_uri=prefix_uri,
            batch_size=dataset_config.batch_size,
            split=str(dataset_config.split),
            annotations_uri=str(dataset_config.annotations_uri),
            transform_name=dataset_config.transform_name,
            text_transform_name=dataset_config.text_transform_name,
            drop_last=dataset_config.drop_last,
            shuffle=dataset_config.shuffle,
            seed=dataset_config.seed,
            force_rebuild=force_rebuild,
        )

    raise ValueError(
        f"Unsupported dataset_format={dataset_config.dataset_format!r} "
        f"for dataset {dataset_config.dataset_id!r}"
    )


def build_synthetic_torch_dataset(
    *,
    dataset_id: str = "synthetic-torch-dataset",
    num_samples: int = 100,
    prefix_uri: str = "memory://synthetic-torch",
    batch_size: int,
    input_shape: tuple[int, ...] = (3, 32, 32),
    num_classes: int = 10,
    transform_name: str | None = None,
    drop_last: bool = False,
    shuffle: bool = True,
    seed: int = 0,
) -> Dataset:
    if not input_shape:
        raise ValueError("input_shape must be non-empty")

    if any(dim <= 0 for dim in input_shape):
        raise ValueError(f"input_shape dimensions must be > 0, got {input_shape}")

    if num_classes <= 0:
        raise ValueError(f"num_classes must be > 0, got {num_classes}")

    return _build_synthetic_dataset(
        dataset_id=dataset_id,
        num_samples=num_samples,
        prefix_uri=prefix_uri,
        batch_size=batch_size,
        transform_name=transform_name,
        drop_last=drop_last,
        shuffle=shuffle,
        seed=seed,
        dataset_format="synthetic_torch",
        payload_format=PayloadFormat.TORCH_BATCH,
        extra_metadata={
            "input_shape": list(input_shape),
            "num_classes": int(num_classes),
        },
    )


def _build_synthetic_dataset(
    *,
    dataset_id: str,
    num_samples: int,
    prefix_uri: str,
    batch_size: int,
    transform_name: str | None,
    drop_last: bool,
    shuffle: bool,
    seed: int,
    dataset_format: str,
    payload_format: PayloadFormat,
    extra_metadata: dict[str, Any],
) -> Dataset:
    if num_samples <= 0:
        raise ValueError(f"num_samples must be > 0, got {num_samples}")

    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    normalized_prefix_uri = prefix_uri.rstrip("/")
    samples = [
        Sample(
            sample_id=str(index),
            source_uri=f"{normalized_prefix_uri}/sample-{index}",
            transform_name=transform_name,
        )
        for index in range(num_samples)
    ]

    return Dataset(
        dataset_id=dataset_id,
        samples=samples,
        batch_size=batch_size,
        payload_format=payload_format,
        dataset_format=dataset_format,
        drop_last=drop_last,
        shuffle=shuffle,
        seed=seed,
        transform_name=transform_name,
        metadata={
            "source": "synthetic",
            "storage": "memory",
            "num_samples": len(samples),
            "prefix_uri": normalized_prefix_uri,
            **extra_metadata,
        },
    )



def build_s3_coco_retrieval_dataset(
    *,
    dataset_id: str,
    prefix_uri: str,
    batch_size: int,
    split: str,
    annotations_uri: str,
    transform_name: str | None = None,
    text_transform_name: str | None = None,
    drop_last: bool = True,
    shuffle: bool = True,
    seed: int = 0,
    manifest_uri: str | None = None,
    use_cached_manifest: bool = True,
    write_manifest_if_missing: bool = True,
    force_rebuild: bool = False,
) -> Dataset:
    """Build the COCO image-text retrieval dataset used by W3.

    The preferred input is the ALBEF COCO Karpathy training JSON. Each item
    has ``image``, ``caption``, and ``image_id`` fields. ``prefix_uri`` points
    at the common COCO image root containing both ``train2014/`` and
    ``val2014/``. The image path from the JSON is joined underneath that root.

    For compatibility with older artifact configs, the standard COCO captions
    JSON format (``images`` + ``annotations``) is also accepted.

    ``Sample.class_name`` carries the caption and ``Sample.label`` carries a
    contiguous image identifier. All captions for the same image therefore use
    the same identifier, matching ALBEF's retrieval training dataset.
    """
    base = prefix_uri.rstrip("/")
    manifest_uri = manifest_uri or _default_manifest_uri(
        prefix_uri=prefix_uri,
        dataset_id=dataset_id,
        split=split,
    )

    if use_cached_manifest and not force_rebuild:
        cached = _load_dataset_from_s3(
            manifest_uri=manifest_uri,
            batch_size=batch_size,
            transform_name=transform_name,
            drop_last=drop_last,
            shuffle=shuffle,
            seed=seed,
            default_payload_format=PayloadFormat.TORCH_BATCH,
            default_dataset_format="coco_retrieval",
        )
        if cached is not None:
            cached.metadata["text_transform_name"] = (
                text_transform_name or cached.metadata.get("text_transform_name", "")
            )
            return cached

    payload = read_s3_json(annotations_uri)

    samples: list[Sample] = []
    image_to_idx: dict[str, int] = {}

    if isinstance(payload, list):
        # ALBEF/Karpathy format used by the original ALBEF retrieval code:
        # {"image": "train2014/...jpg",
        #  "caption": "...",
        #  "image_id": ...}
        for row_index, annotation in enumerate(payload):
            if not isinstance(annotation, dict):
                continue

            image_path = str(annotation.get("image", "")).strip()
            caption = str(annotation.get("caption", "")).strip()
            raw_image_id = annotation.get("image_id")

            if not image_path or not caption or raw_image_id is None:
                continue

            image_id = str(raw_image_id)
            if image_id not in image_to_idx:
                image_to_idx[image_id] = len(image_to_idx)

            if image_path.startswith("s3://"):
                source_uri = image_path
            else:
                source_uri = f"{base}/{image_path.lstrip('/')}"

            samples.append(
                Sample(
                    sample_id=f"{dataset_id}:{split}:{row_index}",
                    source_uri=source_uri,
                    label=image_to_idx[image_id],
                    class_name=caption,
                    transform_name=transform_name,
                )
            )

        annotation_format = "albef_karpathy"

    elif isinstance(payload, dict):
        # Backward-compatible support for the standard COCO captions format.
        images = payload.get("images", [])
        annotations = payload.get("annotations", [])
        if not isinstance(images, list) or not isinstance(annotations, list):
            raise ValueError(
                f"COCO annotation file {annotations_uri} must contain either "
                "an ALBEF/Karpathy list or images/annotations lists"
            )

        image_files: dict[int, str] = {}
        for image in images:
            if not isinstance(image, dict):
                continue
            try:
                image_id = int(image["id"])
                file_name = str(image["file_name"]).strip()
            except (KeyError, TypeError, ValueError):
                continue
            if file_name:
                image_files[image_id] = file_name

        for row_index, annotation in enumerate(annotations):
            if not isinstance(annotation, dict):
                continue

            try:
                image_id_int = int(annotation["image_id"])
                caption = str(annotation["caption"]).strip()
            except (KeyError, TypeError, ValueError):
                continue

            file_name = image_files.get(image_id_int)
            if not file_name or not caption:
                continue

            image_id = str(image_id_int)
            if image_id not in image_to_idx:
                image_to_idx[image_id] = len(image_to_idx)

            samples.append(
                Sample(
                    sample_id=f"{dataset_id}:{split}:{row_index}",
                    source_uri=f"{base}/{file_name.lstrip('/')}",
                    label=image_to_idx[image_id],
                    class_name=caption,
                    transform_name=transform_name,
                )
            )

        annotation_format = "coco_captions"

    else:
        raise ValueError(
            f"COCO annotation file must contain a JSON list or object: {annotations_uri}"
        )

    if not samples:
        raise ValueError(f"no COCO retrieval samples found in {annotations_uri}")

    dataset = Dataset(
        dataset_id=dataset_id,
        samples=samples,
        batch_size=batch_size,
        payload_format=PayloadFormat.TORCH_BATCH,
        dataset_format="coco_retrieval",
        drop_last=drop_last,
        shuffle=shuffle,
        seed=seed,
        transform_name=transform_name,
        metadata={
            "storage": "s3",
            "prefix_uri": prefix_uri,
            "split": split,
            "annotations_uri": annotations_uri,
            "annotation_format": annotation_format,
            "text_transform_name": text_transform_name or "albef_text_30",
            "num_samples": len(samples),
            "num_images": len(image_to_idx),
            "manifest_uri": manifest_uri,
        },
    )

    LOGGER.info(
        "COCO retrieval index ready split=%s format=%s samples=%d images=%d",
        split,
        annotation_format,
        len(samples),
        len(image_to_idx),
    )
    _maybe_save_manifest(
        dataset=dataset,
        manifest_uri=manifest_uri,
        enabled=write_manifest_if_missing,
    )
    return dataset


def build_s3_imagefolder_dataset(
    *,
    dataset_id: str,
    prefix_uri: str,
    batch_size: int,
    split: str = "train",
    extensions: tuple[str, ...] = IMAGE_EXTENSIONS,
    transform_name: str | None = None,
    drop_last: bool = False,
    shuffle: bool = True,
    seed: int = 0,
    manifest_uri: str | None = None,
    use_cached_manifest: bool = True,
    write_manifest_if_missing: bool = True,
    force_rebuild: bool = False,
) -> Dataset:
    base = prefix_uri.rstrip("/")
    split_prefix = f"{base}/{split}/"
    manifest_uri = manifest_uri or _default_manifest_uri(
        prefix_uri=prefix_uri,
        dataset_id=dataset_id,
        split=split,
    )

    if use_cached_manifest and not force_rebuild:
        cached = _load_dataset_from_s3(
            manifest_uri=manifest_uri,
            batch_size=batch_size,
            transform_name=transform_name,
            drop_last=drop_last,
            shuffle=shuffle,
            seed=seed,
            default_payload_format=PayloadFormat.TORCH_BATCH,
            default_dataset_format="image_folder",
        )
        if cached is not None:
            return cached

    extensions_lower = tuple(ext.lower() for ext in extensions)
    uri_class_pairs: list[tuple[str, str]] = []
    class_names: set[str] = set()

    for uri in iter_s3_prefix(split_prefix):
        if not uri.lower().endswith(extensions_lower) or not uri.startswith(split_prefix):
            continue

        relative_path = uri[len(split_prefix) :]
        slash_index = relative_path.find("/")
        if slash_index <= 0:
            continue

        class_name = relative_path[:slash_index].strip()
        if not class_name:
            continue

        class_names.add(class_name)
        uri_class_pairs.append((uri, class_name))

    if not uri_class_pairs:
        raise ValueError(
            f"no image files found under {split_prefix} with extensions={extensions}"
        )

    sorted_class_names = sorted(class_names)
    class_to_idx = {
        class_name: index for index, class_name in enumerate(sorted_class_names)
    }
    samples = [
        Sample(
            sample_id=f"{dataset_id}:{split}:{index}",
            source_uri=uri,
            label=class_to_idx[class_name],
            class_name=class_name,
            transform_name=transform_name,
        )
        for index, (uri, class_name) in enumerate(uri_class_pairs)
    ]

    dataset = Dataset(
        dataset_id=dataset_id,
        samples=samples,
        batch_size=batch_size,
        payload_format=PayloadFormat.TORCH_BATCH,
        dataset_format="image_folder",
        drop_last=drop_last,
        shuffle=shuffle,
        seed=seed,
        transform_name=transform_name,
        metadata={
            "storage": "s3",
            "prefix_uri": prefix_uri,
            "split": split,
            "num_samples": len(samples),
            "num_classes": len(class_to_idx),
            "class_names": sorted_class_names,
            "class_to_idx": class_to_idx,
            "manifest_uri": manifest_uri,
        },
    )

    _maybe_save_manifest(
        dataset=dataset,
        manifest_uri=manifest_uri,
        enabled=write_manifest_if_missing,
    )
    return dataset


def build_s3_openimages_dataset(
    *,
    dataset_id: str,
    prefix_uri: str,
    batch_size: int,
    split: str,
    annotations_uri: str,
    class_descriptions_uri: str | None,
    transform_name: str | None = None,
    drop_last: bool = False,
    shuffle: bool = True,
    seed: int = 0,
    manifest_uri: str | None = None,
    use_cached_manifest: bool = True,
    write_manifest_if_missing: bool = True,
    force_rebuild: bool = False,
) -> Dataset:
    """Build the Open Images W2 single-label projection used by the artifact.

    Open Images stores images flat as <split>/<ImageID>.jpg and keeps labels in
    a separate CSV. A logical sample is created for each positive
    (ImageID, LabelName) annotation. This preserves the behavior of the older
    experiment code while allowing the current classification training loop to
    keep using CrossEntropyLoss.
    """
    base = prefix_uri.rstrip("/")
    manifest_uri = manifest_uri or _default_manifest_uri(
        prefix_uri=prefix_uri,
        dataset_id=dataset_id,
        split=split,
    )

    if use_cached_manifest and not force_rebuild:
        cached = _load_dataset_from_s3(
            manifest_uri=manifest_uri,
            batch_size=batch_size,
            transform_name=transform_name,
            drop_last=drop_last,
            shuffle=shuffle,
            seed=seed,
            default_payload_format=PayloadFormat.TORCH_BATCH,
            default_dataset_format="openimages_boxable",
        )
        if cached is not None:
            return cached

    class_display_names = (
        _load_openimages_class_descriptions(class_descriptions_uri)
        if class_descriptions_uri
        else {}
    )

    if class_display_names:
        label_names = sorted(class_display_names)
        class_to_idx = {label_name: index for index, label_name in enumerate(label_names)}
        samples = _load_openimages_positive_samples(
            dataset_id=dataset_id,
            base_uri=base,
            split=split,
            annotations_uri=annotations_uri,
            transform_name=transform_name,
            class_to_idx=class_to_idx,
        )
    else:
        rows = list(_iter_positive_openimages_annotations(annotations_uri))
        label_names = sorted({label_name for _, label_name in rows})
        class_to_idx = {label_name: index for index, label_name in enumerate(label_names)}
        samples = _samples_from_openimages_rows(
            dataset_id=dataset_id,
            base_uri=base,
            split=split,
            rows=rows,
            transform_name=transform_name,
            class_to_idx=class_to_idx,
        )

    if not samples:
        raise ValueError(
            f"no positive Open Images samples found in annotations {annotations_uri}"
        )

    dataset = Dataset(
        dataset_id=dataset_id,
        samples=samples,
        batch_size=batch_size,
        payload_format=PayloadFormat.TORCH_BATCH,
        dataset_format="openimages_boxable",
        drop_last=drop_last,
        shuffle=shuffle,
        seed=seed,
        transform_name=transform_name,
        metadata={
            "storage": "s3",
            "prefix_uri": prefix_uri,
            "split": split,
            "annotations_uri": annotations_uri,
            "class_descriptions_uri": class_descriptions_uri or "",
            "num_samples": len(samples),
            "num_classes": len(class_to_idx),
            "class_names": label_names,
            "class_to_idx": class_to_idx,
            "class_display_names": class_display_names,
            "manifest_uri": manifest_uri,
            "label_projection": "positive_image_label_pairs",
        },
    )

    LOGGER.info(
        "Open Images index ready split=%s samples=%d classes=%d",
        split,
        len(samples),
        len(class_to_idx),
    )
    _maybe_save_manifest(
        dataset=dataset,
        manifest_uri=manifest_uri,
        enabled=write_manifest_if_missing,
    )
    return dataset


def _load_openimages_positive_samples(
    *,
    dataset_id: str,
    base_uri: str,
    split: str,
    annotations_uri: str,
    transform_name: str | None,
    class_to_idx: dict[str, int],
) -> list[Sample]:
    samples: list[Sample] = []

    for row_index, (image_id, label_name) in enumerate(
        _iter_positive_openimages_annotations(annotations_uri)
    ):
        label = class_to_idx.get(label_name)
        if label is None:
            continue

        samples.append(
            Sample(
                sample_id=f"{dataset_id}:{split}:{row_index}",
                source_uri=f"{base_uri}/{split}/{image_id}.jpg",
                label=label,
                class_name=label_name,
                transform_name=transform_name,
            )
        )

    return samples


def _samples_from_openimages_rows(
    *,
    dataset_id: str,
    base_uri: str,
    split: str,
    rows: list[tuple[str, str]],
    transform_name: str | None,
    class_to_idx: dict[str, int],
) -> list[Sample]:
    return [
        Sample(
            sample_id=f"{dataset_id}:{split}:{index}",
            source_uri=f"{base_uri}/{split}/{image_id}.jpg",
            label=class_to_idx[label_name],
            class_name=label_name,
            transform_name=transform_name,
        )
        for index, (image_id, label_name) in enumerate(rows)
    ]


def _iter_positive_openimages_annotations(
    annotations_uri: str,
) -> Iterator[tuple[str, str]]:
    with _open_s3_csv(annotations_uri) as reader:
        fieldnames = set(reader.fieldnames or ())
        required = {"ImageID", "LabelName", "Confidence"}
        missing = required - fieldnames
        if missing:
            raise ValueError(
                f"Open Images annotation CSV {annotations_uri} is missing "
                f"columns {sorted(missing)}"
            )

        for row in reader:
            image_id = str(row.get("ImageID", "")).strip()
            label_name = str(row.get("LabelName", "")).strip()
            confidence = str(row.get("Confidence", "")).strip()

            if not image_id or not label_name:
                continue

            try:
                is_positive = float(confidence) == 1.0
            except ValueError:
                is_positive = False

            if is_positive:
                yield image_id, label_name


def _load_openimages_class_descriptions(uri: str) -> dict[str, str]:
    descriptions: dict[str, str] = {}

    with _open_s3_csv_rows(uri) as rows:
        for row in rows:
            if len(row) < 2:
                continue
            label_name = row[0].strip()
            display_name = row[1].strip()
            if label_name:
                descriptions[label_name] = display_name

    if not descriptions:
        raise ValueError(f"no Open Images class descriptions found in {uri}")

    return descriptions


class _S3CsvDictContext:
    def __init__(self, uri: str) -> None:
        self.uri = uri
        self.body = None
        self.text = None
        self.reader = None

    def __enter__(self) -> csv.DictReader:
        loc = parse_s3_uri(self.uri)
        response = get_s3_client().get_object(Bucket=loc.bucket, Key=loc.key)
        self.body = response["Body"]
        self.text = io.TextIOWrapper(self.body, encoding="utf-8", newline="")
        self.reader = csv.DictReader(self.text)
        return self.reader

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.text is not None:
            self.text.close()
        elif self.body is not None:
            self.body.close()


class _S3CsvRowsContext:
    def __init__(self, uri: str) -> None:
        self.uri = uri
        self.body = None
        self.text = None
        self.reader = None

    def __enter__(self) -> csv.reader:
        loc = parse_s3_uri(self.uri)
        response = get_s3_client().get_object(Bucket=loc.bucket, Key=loc.key)
        self.body = response["Body"]
        self.text = io.TextIOWrapper(self.body, encoding="utf-8", newline="")
        self.reader = csv.reader(self.text)
        return self.reader

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.text is not None:
            self.text.close()
        elif self.body is not None:
            self.body.close()


def _open_s3_csv(uri: str) -> _S3CsvDictContext:
    return _S3CsvDictContext(uri)


def _open_s3_csv_rows(uri: str) -> _S3CsvRowsContext:
    return _S3CsvRowsContext(uri)


def _is_synthetic_torch_memory_uri(prefix_uri: str) -> bool:
    return (
        prefix_uri == "memory://synthetic-torch"
        or prefix_uri.startswith("memory://synthetic-torch/")
    )


def _default_manifest_uri(
    *,
    prefix_uri: str,
    dataset_id: str,
    split: str,
) -> str:
    base = prefix_uri.rstrip("/")
    return f"{base}/batchflow/manifests/{dataset_id}_{split}.manifest.json"


def _load_dataset_from_s3(
    *,
    manifest_uri: str,
    batch_size: int,
    transform_name: str | None,
    drop_last: bool,
    shuffle: bool,
    seed: int,
    default_payload_format: PayloadFormat,
    default_dataset_format: str,
) -> Dataset | None:
    try:
        payload = read_s3_json(manifest_uri)
    except Exception as exc:
        LOGGER.info(
            "no cached manifest loaded from %s (%s)",
            manifest_uri,
            type(exc).__name__,
        )
        return None

    if not isinstance(payload, dict):
        LOGGER.warning("manifest at %s is not a JSON object", manifest_uri)
        return None

    try:
        dataset = _dataset_from_payload(
            payload,
            batch_size=batch_size,
            transform_name=transform_name,
            drop_last=drop_last,
            shuffle=shuffle,
            seed=seed,
            default_payload_format=default_payload_format,
            default_dataset_format=default_dataset_format,
        )
        if dataset.dataset_format != default_dataset_format:
            LOGGER.warning(
                "ignoring manifest %s with dataset_format=%s; expected %s",
                manifest_uri,
                dataset.dataset_format,
                default_dataset_format,
            )
            return None
        return dataset
    except Exception:
        LOGGER.exception("failed parsing manifest payload from %s", manifest_uri)
        return None


def _dataset_from_payload(
    payload: dict[str, Any],
    *,
    batch_size: int,
    transform_name: str | None,
    drop_last: bool,
    shuffle: bool,
    seed: int,
    default_payload_format: PayloadFormat,
    default_dataset_format: str,
) -> Dataset:
    metadata = dict(payload.get("metadata", {}))
    payload_format = _payload_format_from_payload(
        payload,
        default=default_payload_format,
    )
    dataset_format = str(payload.get("dataset_format") or default_dataset_format)

    samples = [
        Sample(
            sample_id=str(item["sample_id"]),
            source_uri=str(item["source_uri"]),
            label=int(item["label"]) if item.get("label") is not None else None,
            class_name=(
                str(item["class_name"])
                if item.get("class_name") is not None
                else None
            ),
            transform_name=transform_name,
        )
        for item in payload.get("samples", [])
    ]
    metadata["num_samples"] = len(samples)

    return Dataset(
        dataset_id=str(payload["dataset_id"]),
        samples=samples,
        batch_size=batch_size,
        payload_format=payload_format,
        dataset_format=dataset_format,
        drop_last=drop_last,
        shuffle=shuffle,
        seed=seed,
        transform_name=transform_name,
        metadata=metadata,
    )


def _maybe_save_manifest(
    *,
    dataset: Dataset,
    manifest_uri: str,
    enabled: bool,
) -> None:
    if not enabled:
        return

    try:
        _save_dataset_manifest_to_s3(dataset=dataset, manifest_uri=manifest_uri)
    except Exception:
        LOGGER.exception("failed to save manifest to %s", manifest_uri)


def _save_dataset_manifest_to_s3(
    *,
    dataset: Dataset,
    manifest_uri: str,
) -> None:
    payload = {
        "dataset_id": dataset.dataset_id,
        "dataset_format": dataset.dataset_format,
        "payload_format": dataset.payload_format.value,
        "metadata": dict(dataset.metadata),
        "samples": [
            {
                "sample_id": sample.sample_id,
                "source_uri": sample.source_uri,
                "label": sample.label,
                "class_name": sample.class_name,
            }
            for sample in dataset.samples
        ],
    }
    write_s3_json(manifest_uri, payload)


def _payload_format_from_payload(
    payload: dict[str, Any],
    *,
    default: PayloadFormat,
) -> PayloadFormat:
    value = payload.get("payload_format")
    if not value:
        return default
    return PayloadFormat(str(value))
