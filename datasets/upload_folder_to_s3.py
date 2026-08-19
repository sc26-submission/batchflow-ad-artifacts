import os
import argparse
from concurrent.futures import ThreadPoolExecutor
import boto3


def upload_folder(local_dir, bucket, prefix="", workers=8):
    s3 = boto3.client("s3")

    def upload_file(file_path):
        rel_path = os.path.relpath(file_path, local_dir)
        s3_key = os.path.join(prefix, rel_path).replace("\\", "/")
        s3.upload_file(file_path, bucket, s3_key)

    print(f"Uploading {local_dir} → s3://{bucket}/{prefix}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for root, _, files in os.walk(local_dir):
            for file in files:
                full_path = os.path.join(root, file)
                executor.submit(upload_file, full_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", required=True, help="Path to local folder")
    parser.add_argument("--s3_bucket", required=True, help="S3 bucket name")
    parser.add_argument("--s3_prefix", default="", help="S3 prefix (folder)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    upload_folder(args.local_dir, args.s3_bucket, args.s3_prefix, args.workers)


if __name__ == "__main__":
    main()