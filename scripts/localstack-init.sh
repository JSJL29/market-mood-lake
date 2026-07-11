#!/bin/sh
set -eu

awslocal s3api head-bucket --bucket raw >/dev/null 2>&1 || awslocal s3 mb s3://raw

SOURCE=/bootstrap/SP500_Historical_Data.csv
if [ -s "$SOURCE" ]; then
    if ! awslocal s3api head-object --bucket raw --key sp500_combined.csv >/dev/null 2>&1; then
        awslocal s3 cp "$SOURCE" s3://raw/sp500_combined.csv
    fi
fi
