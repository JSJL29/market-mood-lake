"""Create the empty Raw bucket without depending on a host-side CSV."""

import boto3
from botocore.exceptions import ClientError


s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:4566",
    aws_access_key_id="test",
    aws_secret_access_key="test",
    region_name="us-east-1",
)
try:
    s3.head_bucket(Bucket="raw")
except ClientError:
    s3.create_bucket(Bucket="raw")
