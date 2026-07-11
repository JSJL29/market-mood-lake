import argparse
import json
import statistics
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests


def make_payload(batch_size: int) -> dict:
    rows = []

    for i in range(batch_size):
        rows.append(
            {
                "date": (date(2090, 1, 1) + timedelta(days=i)).isoformat(),
                "close": 100.0 + i,
                "volume": 1_000_000 + i,
                "vix_close": 18.5,
            }
        )

    return {"data": rows, "benchmark": True}


def call_endpoint(base_url: str, endpoint: str, batch_size: int, timeout: int) -> float:
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    payload = make_payload(batch_size)

    start = time.perf_counter()
    response = requests.post(url, json=payload, timeout=timeout)
    elapsed = time.perf_counter() - start

    if response.status_code >= 400:
        raise RuntimeError(
            f"{endpoint} batch={batch_size} failed "
            f"status={response.status_code}, body={response.text[:500]}"
        )

    return elapsed


def benchmark(base_url: str, batch_sizes: list[int], runs: int, timeout: int) -> dict:
    results = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "runs": runs,
        "batch_sizes": batch_sizes,
        "results": {},
    }

    # Chauffe Numba, le pool MySQL et la table dédiée avant toute mesure.
    call_endpoint(base_url, "/ingest", max(batch_sizes), timeout)
    call_endpoint(base_url, "/ingest_fast", max(batch_sizes), timeout)

    for batch_size in batch_sizes:
        ingest_times = []
        ingest_fast_times = []

        print(f"\n===== Batch {batch_size} =====")

        for run_idx in range(1, runs + 1):
            t_ingest = call_endpoint(base_url, "/ingest", batch_size, timeout)
            t_fast = call_endpoint(base_url, "/ingest_fast", batch_size, timeout)

            ingest_times.append(t_ingest)
            ingest_fast_times.append(t_fast)

            print(
                f"Run {run_idx}/{runs} | "
                f"/ingest={t_ingest:.4f}s | "
                f"/ingest_fast={t_fast:.4f}s"
            )

        avg_ingest = statistics.mean(ingest_times)
        avg_fast = statistics.mean(ingest_fast_times)
        median_ingest = statistics.median(ingest_times)
        median_fast = statistics.median(ingest_fast_times)

        improvement_avg = ((avg_ingest - avg_fast) / avg_ingest) * 100 if avg_ingest else 0
        improvement_median = ((median_ingest - median_fast) / median_ingest) * 100 if median_ingest else 0

        results["results"][str(batch_size)] = {
            "ingest_times": ingest_times,
            "ingest_fast_times": ingest_fast_times,
            "avg_ingest_seconds": avg_ingest,
            "avg_ingest_fast_seconds": avg_fast,
            "median_ingest_seconds": median_ingest,
            "median_ingest_fast_seconds": median_fast,
            "improvement_avg_percent": improvement_avg,
            "improvement_median_percent": improvement_median,
            "passes_30_percent_requirement": improvement_avg >= 30,
        }

        print("\nRésumé batch", batch_size)
        print(f"Average /ingest      : {avg_ingest:.4f}s")
        print(f"Average /ingest_fast : {avg_fast:.4f}s")
        print(f"Gain moyen           : {improvement_avg:.2f}%")
        print(f"Gain médian          : {improvement_median:.2f}%")
        print(f"Objectif 30%         : {'OK' if improvement_avg >= 30 else 'KO'}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 100])
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--output",
        default="data/benchmarks/ingest_benchmark_results.json",
    )

    args = parser.parse_args()

    results = benchmark(
        base_url=args.base_url,
        batch_sizes=args.batch_sizes,
        runs=args.runs,
        timeout=args.timeout,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nRésultats écrits dans: {output_path}")


if __name__ == "__main__":
    main()
