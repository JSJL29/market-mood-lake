"""Paired, order-balanced benchmark for /ingest and /ingest_fast."""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests


def make_payload(batch_size: int, offset: int = 0) -> dict:
    start = date(2050, 1, 1) + timedelta(days=offset)
    return {
        "benchmark": True,
        "data": [
            {
                "date": (start + timedelta(days=i)).isoformat(),
                "close": 100.0 + i,
                "volume": 1_000_000 + i,
                "vix_close": 18.5,
            }
            for i in range(batch_size)
        ],
    }


def call_endpoint(base_url: str, endpoint: str, payload: dict, timeout: int) -> dict:
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    start = time.perf_counter()
    response = requests.post(url, json=payload, timeout=timeout)
    wall_seconds = time.perf_counter() - start
    if response.status_code >= 400:
        raise RuntimeError(f"{endpoint} failed status={response.status_code}: {response.text[:500]}")
    body = response.json()
    if body.get("batch_size") != len(payload["data"]):
        raise RuntimeError(f"Unexpected response from {endpoint}: {body}")
    return {"wall_seconds": wall_seconds, "server_seconds": float(body["elapsed_seconds"])}


def _summary(values: list[float]) -> dict:
    return {
        "mean_seconds": statistics.mean(values),
        "median_seconds": statistics.median(values),
        "min_seconds": min(values),
        "max_seconds": max(values),
    }


def benchmark(
    base_url: str,
    batch_sizes: list[int],
    runs: int,
    timeout: int,
    min_improvement: float = 30.0,
) -> dict:
    if runs < 4:
        raise ValueError("At least 4 paired runs are required")
    health = requests.get(f"{base_url.rstrip('/')}/health", timeout=timeout)
    health.raise_for_status()
    results = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "protocol": "isolated-table-reset-paired-fresh-rows-alternating-order",
        "criterion": f"median paired server-time improvement >= {min_improvement}%",
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
        "runs": runs,
        "batch_sizes": batch_sizes,
        "results": {},
    }

    reset = requests.delete(f"{base_url.rstrip('/')}/ingest/benchmark-data", timeout=timeout)
    reset.raise_for_status()
    results["benchmark_table_reset"] = reset.json()

    # Warm Numba, connection pools and table creation outside measurements.
    for index, endpoint in enumerate(("/ingest", "/ingest_fast")):
        call_endpoint(base_url, endpoint, make_payload(max(batch_sizes), 100 + index * 500), timeout)

    offset = 2_000
    for batch_size in batch_sizes:
        pairs = []
        print(f"\n===== Batch {batch_size}: {runs} paired runs =====")
        for run_index in range(runs):
            order = ("/ingest", "/ingest_fast") if run_index % 2 == 0 else ("/ingest_fast", "/ingest")
            measurements = {}
            # Distinct date ranges make both calls perform fresh INSERT operations.
            for endpoint_index, endpoint in enumerate(order):
                payload = make_payload(batch_size, offset + endpoint_index * (batch_size + 2))
                measurements[endpoint] = call_endpoint(base_url, endpoint, payload, timeout)
            offset += 2 * (batch_size + 2)
            slow = measurements["/ingest"]["server_seconds"]
            fast = measurements["/ingest_fast"]["server_seconds"]
            improvement = ((slow - fast) / slow) * 100 if slow else 0.0
            pairs.append(
                {
                    "run": run_index + 1,
                    "order": list(order),
                    "ingest": measurements["/ingest"],
                    "ingest_fast": measurements["/ingest_fast"],
                    "paired_server_improvement_percent": improvement,
                }
            )
            print(f"Pair {run_index + 1:02d} {order}: {improvement:.2f}%")

        improvements = [pair["paired_server_improvement_percent"] for pair in pairs]
        ingest_server = [pair["ingest"]["server_seconds"] for pair in pairs]
        fast_server = [pair["ingest_fast"]["server_seconds"] for pair in pairs]
        ingest_wall = [pair["ingest"]["wall_seconds"] for pair in pairs]
        fast_wall = [pair["ingest_fast"]["wall_seconds"] for pair in pairs]
        median_improvement = statistics.median(improvements)
        batch_result = {
            "pairs": pairs,
            "server": {"ingest": _summary(ingest_server), "ingest_fast": _summary(fast_server)},
            "wall": {"ingest": _summary(ingest_wall), "ingest_fast": _summary(fast_wall)},
            "paired_improvement": {
                "mean_percent": statistics.mean(improvements),
                "median_percent": median_improvement,
                "positive_pairs": sum(value > 0 for value in improvements),
                "total_pairs": runs,
            },
            "passes_requirement": median_improvement >= min_improvement,
        }
        results["results"][str(batch_size)] = batch_result
        print(
            f"Median paired improvement: {median_improvement:.2f}% - "
            f"{'PASS' if batch_result['passes_requirement'] else 'FAIL'}"
        )
    results["all_requirements_pass"] = all(
        item["passes_requirement"] for item in results["results"].values()
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 100])
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--min-improvement", type=float, default=30.0)
    parser.add_argument("--output", default="data/benchmarks/ingest_benchmark_results.json")
    args = parser.parse_args()
    results = benchmark(args.base_url, args.batch_sizes, args.runs, args.timeout, args.min_improvement)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nResults written to {output}")
    if not results["all_requirements_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
