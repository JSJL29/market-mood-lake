#!/usr/bin/env python3
"""Create the LocalStack S3 raw bucket if it does not already exist."""
import argparse
import os
import sys
import time

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError


def create_bucket(endpoint_url: str, bucket: str, retries: int = 20, sleep_s: float = 1.5) -> None:
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

    s3 = boto3.client("s3", endpoint_url=endpoint_url)
    last_error = None
    for _ in range(retries):
        try:
            existing = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]
            if bucket in existing:
                print(f"Bucket s3://{bucket} déjà présent sur {endpoint_url}")
                return
            s3.create_bucket(Bucket=bucket)
            print(f"Bucket s3://{bucket} créé sur {endpoint_url}")
            return
        except (EndpointConnectionError, ClientError) as exc:
            last_error = exc
            time.sleep(sleep_s)

    print(f"Impossible de créer s3://{bucket} sur {endpoint_url}: {last_error}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-url", default=os.getenv("S3_ENDPOINT_URL", "http://localhost:4566"))
    parser.add_argument("--bucket", default=os.getenv("RAW_BUCKET", "raw"))
    args = parser.parse_args()
    create_bucket(args.endpoint_url, args.bucket)


if __name__ == "__main__":
    main()
