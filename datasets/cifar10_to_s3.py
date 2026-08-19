from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import boto3
from PIL import Image
from torchvision.datasets import CIFAR10


def save_imagefolder(dataset, root: Path, split: str) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)

    classes = dataset.classes

    # create class dirs
    for cls in classes:
        (split_dir / cls).mkdir(parents=True, exist_ok=True)

    for idx, (img, label) in enumerate(dataset):
        cls_name = classes[label]
        out_path = split_dir / cls_name / f"{idx:06d}.png"

        if isinstance(img, Image.Image):
            img.save(out_path)
        else:
            Image.fromarray(img).save(out_path)

        if idx % 5000 == 0:
            print(f"[{split}] saved {idx}/{len(dataset)}")


def upload_directory_to_s3(local_dir: Path, bucket: str, prefix: str) -> None:
    s3 = boto3.client("s3")

    for path in local_dir.rglob("*"):
        if path.is_file():
            rel_path = path.relative_to(local_dir)
            s3_key = f"{prefix}/{rel_path.as_posix()}" if prefix else rel_path.as_posix()

            print(f"Uploading {path} → s3://{bucket}/{s3_key}")
            s3.upload_file(str(path), bucket, s3_key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-bucket",  default="cifar10-bf")
    parser.add_argument("--s3-prefix", default="")
    parser.add_argument("--tmp-dir", default="datasets/tmp_cifar10")
    args = parser.parse_args()

    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else Path(tempfile.mkdtemp())
    print(f"Using temp dir: {tmp_dir}")

    # download CIFAR-10
    train_ds = CIFAR10(root=tmp_dir, train=True, download=True)
    val_ds = CIFAR10(root=tmp_dir, train=False, download=True)

    imagefolder_root = tmp_dir / "cifar10_imagefolder"
    imagefolder_root.mkdir(parents=True, exist_ok=True)

    print("Saving train split...")
    save_imagefolder(train_ds, imagefolder_root, "train")

    print("Saving val split...")
    save_imagefolder(val_ds, imagefolder_root, "val")

    print("Uploading to S3...")
    upload_directory_to_s3(
        imagefolder_root,
        bucket=args.s3_bucket,
        prefix=args.s3_prefix,
    )

    print(f"Cleaning up temp dir: {tmp_dir}")
    for path in tmp_dir.rglob("*"):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    

    print("Done!")


if __name__ == "__main__":
    main()