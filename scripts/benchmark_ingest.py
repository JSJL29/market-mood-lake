"""
Benchmark /ingest vs /ingest_fast, conformément au niveau avancé du sujet
final. Mesure le temps d'exécution pour un batch de 1 et un batch de 100
éléments, avec plusieurs essais pour lisser le bruit (le premier appel à
/ingest_fast inclut la compilation JIT de Numba, donc on l'exclut de la
moyenne comme "warm-up").

Usage:
    python scripts/benchmark_ingest.py --api_url http://localhost:8000
"""
import argparse
import statistics
from datetime import datetime, timedelta

import requests


def build_batch(size, start_date="2020-01-01"):
    """Génère un batch synthétique de `size` MarketTick."""
    base_date = datetime.strptime(start_date, "%Y-%m-%d")
    batch = []
    close = 3000.0
    for i in range(size):
        close += (i % 7) - 3  # petite variation, sans importance pour le timing
        batch.append({
            "date": (base_date + timedelta(days=i)).strftime("%Y-%m-%d"),
            "close": round(close, 2),
            "volume": 1_000_000 + i * 1000,
            "vix_close": round(15 + (i % 10), 2),
        })
    return batch


def call_endpoint(api_url, endpoint, batch, n_trials=5, warmup=1):
    """Appelle l'endpoint n_trials fois, retourne la liste des temps mesurés côté serveur."""
    url = f"{api_url}/{endpoint}"
    times = []
    for trial in range(warmup + n_trials):
        response = requests.post(url, json={"data": batch}, timeout=30)
        response.raise_for_status()
        elapsed = response.json()["elapsed_seconds"]
        if trial >= warmup:
            times.append(elapsed)
        print(f"  [{endpoint}] batch={len(batch)} essai {trial + 1}: {elapsed:.6f}s"
              + ("  (warm-up, exclu)" if trial < warmup else ""))
    return times


def main():
    parser = argparse.ArgumentParser(description="Benchmark /ingest vs /ingest_fast")
    parser.add_argument("--api_url", type=str, default="http://localhost:8000")
    parser.add_argument("--n_trials", type=int, default=5, help="Nombre d'essais moyennés")
    args = parser.parse_args()

    results = {}

    for batch_size in [1, 100]:
        batch = build_batch(batch_size)
        results[batch_size] = {}

        for endpoint in ["ingest", "ingest_fast"]:
            print(f"\n--- {endpoint} / batch={batch_size} ---")
            warmup = 1 if endpoint == "ingest_fast" else 0
            times = call_endpoint(args.api_url, endpoint, batch, args.n_trials, warmup)
            results[batch_size][endpoint] = {
                "mean": statistics.mean(times),
                "stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
            }

    print("\n" + "=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)
    print(f"{'Batch':<10}{'/ingest (s)':<18}{'/ingest_fast (s)':<20}{'Gain':<10}")
    for batch_size, data in results.items():
        naive = data["ingest"]["mean"]
        fast = data["ingest_fast"]["mean"]
        gain = (1 - fast / naive) * 100 if naive > 0 else 0
        print(f"{batch_size:<10}{naive:<18.6f}{fast:<20.6f}{gain:>6.1f}%")

    print("\nCopiez ce tableau dans le README (section niveau avancé).")


if __name__ == "__main__":
    main()