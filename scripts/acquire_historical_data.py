"""Acquire the historical CSV used by the raw ingestion stage.

The command is idempotent: an existing non-empty file is kept.  In CI or on a
fresh machine, ``--source-url`` (or ``HISTORICAL_DATA_URL``) makes acquisition
fully automatic without baking credentials or a vendor-specific URL into Git.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
import os
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen


def _validate_csv(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Historical dataset is missing or empty: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        normalized = {column.strip().lower().replace(" ", "_") for column in header}
        required = {"ticker", "date", "close"}
        if not required.issubset(normalized):
            raise ValueError(f"Historical CSV missing columns: {sorted(required - normalized)}")
        if next(reader, None) is None:
            raise ValueError("Historical CSV contains no data rows")


def _write_demo_dataset(destination: Path, days: int = 520) -> None:
    """Create a deterministic, offline-capable dataset for evaluation/dev."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    start = date(2022, 1, 3)
    tickers = ("SPY", "QQQ", "DIA")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ticker", "date", "open", "high", "low", "close", "volume"])
        for ticker_index, ticker in enumerate(tickers):
            price = 100.0 + 20 * ticker_index
            written = 0
            current = start
            while written < days:
                if current.weekday() < 5:
                    drift = 0.0004 + (((written + ticker_index * 7) % 17) - 8) / 10000
                    open_price = price
                    price *= 1 + drift
                    writer.writerow(
                        [ticker, current.isoformat(), f"{open_price:.4f}", f"{price * 1.003:.4f}",
                         f"{price * 0.997:.4f}", f"{price:.4f}", 1_000_000 + written * 100]
                    )
                    written += 1
                current += timedelta(days=1)
    _validate_csv(temporary)
    temporary.replace(destination)


def acquire(
    destination: Path,
    source_url: Optional[str] = None,
    allow_demo_fallback: bool = True,
) -> str:
    if destination.is_file() and destination.stat().st_size > 0:
        _validate_csv(destination)
        return "existing"
    if not source_url and allow_demo_fallback:
        _write_demo_dataset(destination)
        return "generated-demo"
    if not source_url:
        raise FileNotFoundError(
            f"Historical dataset missing: {destination}. Set HISTORICAL_DATA_URL "
            "to an HTTP(S) CSV URL or place the file at that path."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(source_url, headers={"User-Agent": "market-mood-lake/1.0"})
    try:
        with urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            content_type = response.headers.get_content_type()
            if content_type not in {"text/csv", "text/plain", "application/octet-stream"}:
                raise ValueError(f"Unexpected historical dataset content type: {content_type}")
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if temporary.stat().st_size == 0:
            raise ValueError("Downloaded historical dataset is empty")
        _validate_csv(temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return "downloaded"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--source-url", default=os.getenv("HISTORICAL_DATA_URL"))
    parser.add_argument(
        "--no-demo-fallback", action="store_true",
        help="Fail instead of generating the deterministic offline demo dataset",
    )
    args = parser.parse_args()
    print(
        f"Historical dataset: {acquire(args.destination, args.source_url, not args.no_demo_fallback)}"
    )


if __name__ == "__main__":
    main()
