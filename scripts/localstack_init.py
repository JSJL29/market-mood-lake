"""LocalStack READY hook; Python avoids Windows CRLF/shebang issues."""
from pathlib import Path

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

source = Path("/bootstrap/SP500_Historical_Data.csv")
if source.is_file() and source.stat().st_size:
    try:
        s3.head_object(Bucket="raw", Key="sp500_combined.csv")
    except ClientError:
        s3.upload_file(str(source), "raw", "sp500_combined.csv")
