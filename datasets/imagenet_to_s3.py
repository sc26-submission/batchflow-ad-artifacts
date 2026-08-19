import os
import tarfile
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor
import boto3

# --------- Helpers ---------

def extract_tar(tar_path, dest):
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall(dest)


def extract_train(train_tar, out_dir):
    tmp_dir = os.path.join(out_dir, "train_raw")
    extract_tar(train_tar, tmp_dir)

    train_out = os.path.join(out_dir, "train")
    os.makedirs(train_out, exist_ok=True)

    print("Processing train set...")
    for tar_name in os.listdir(tmp_dir):
        if not tar_name.endswith(".tar"):
            continue

        synset = tar_name.replace(".tar", "")
        synset_dir = os.path.join(train_out, synset)
        os.makedirs(synset_dir, exist_ok=True)

        tar_path = os.path.join(tmp_dir, tar_name)
        with tarfile.open(tar_path) as tar:
            tar.extractall(synset_dir)

    shutil.rmtree(tmp_dir)

def prepare_val(val_tar, out_dir, val_map_file):
    tmp_dir = os.path.join(out_dir, "val_raw")
    extract_tar(val_tar, tmp_dir)

    val_out = os.path.join(out_dir, "val")
    os.makedirs(val_out, exist_ok=True)

    # Each line is the synset label for:
    # ILSVRC2012_val_00000001.JPEG, ..., ILSVRC2012_val_00050000.JPEG
    with open(val_map_file) as f:
        synsets = [line.strip() for line in f if line.strip()]

    print("Processing validation set...")

    for idx, synset in enumerate(synsets, start=1):
        filename = f"ILSVRC2012_val_{idx:08d}.JPEG"

        src = os.path.join(tmp_dir, filename)
        dst_dir = os.path.join(val_out, synset)
        dst = os.path.join(dst_dir, filename)

        os.makedirs(dst_dir, exist_ok=True)

        if not os.path.exists(src):
            print(f"Warning: missing {filename}")
            continue

        shutil.move(src, dst)

    shutil.rmtree(tmp_dir)

def upload_to_s3(local_dir, bucket, prefix, workers=8):
    s3 = boto3.client("s3")

    def upload_file(file_path):
        rel_path = os.path.relpath(file_path, local_dir)
        s3_key = os.path.join(prefix, rel_path)
        s3.upload_file(file_path, bucket, s3_key)

    print("Uploading to S3...")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for root, _, files in os.walk(local_dir):
            for file in files:
                full_path = os.path.join(root, file)
                executor.submit(upload_file, full_path)


# --------- Main ---------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_tar", required=True)
    parser.add_argument("--val_tar", required=True)
    parser.add_argument("--val_map", required=True,
                        help="File mapping val images to synsets")
    parser.add_argument("--output_dir", default="./imagenet")
    parser.add_argument("--s3_bucket", required=True)
    parser.add_argument("--s3_prefix", default="imagenet")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    extract_train(args.train_tar, args.output_dir)
    # prepare_val(args.val_tar, args.output_dir, args.val_map)

    # upload_to_s3(args.output_dir, args.s3_bucket, args.s3_prefix)

    print("Done!")


if __name__ == "__main__":
    main()