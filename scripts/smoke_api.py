#!/usr/bin/env python3
"""Small API smoke test for the correction/demo flow."""
import argparse
import json
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

REQUIRED_ENDPOINTS = ["/health", "/raw/", "/staging/", "/curated/", "/stats"]


def get_json(url: str, timeout: float = 5.0):
    with urlopen(url, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--retries", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=1.5)
    args = parser.parse_args()

    api_url = args.api_url.rstrip("/")
    last_error = None
    for attempt in range(1, args.retries + 1):
        try:
            health = get_json(f"{api_url}/health")
            print("/health OK")
            print(json.dumps(health, indent=2, ensure_ascii=False))
            break
        except (HTTPError, URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
            time.sleep(args.sleep)
    else:
        print(f"API indisponible après {args.retries} essais: {last_error}", file=sys.stderr)
        sys.exit(1)

    ok = True
    for path in REQUIRED_ENDPOINTS[1:]:
        try:
            payload = get_json(f"{api_url}{path}")
            preview = payload if isinstance(payload, dict) else {"items": len(payload or [])}
            print(f"{path} OK -> {json.dumps(preview, ensure_ascii=False)[:500]}")
        except HTTPError as exc:
            # Staging/curated can be empty before ingestion; 500 still signals a useful issue.
            ok = False
            print(f"{path} ERREUR HTTP {exc.code}: {exc.reason}", file=sys.stderr)
        except Exception as exc:
            ok = False
            print(f"{path} ERREUR: {exc}", file=sys.stderr)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
