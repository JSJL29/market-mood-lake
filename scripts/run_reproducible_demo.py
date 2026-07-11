"""Run the evaluator demonstration and emit an auditable JSON report (WSL/Linux)."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(label: str, command: list[str], steps: list[dict]) -> None:
    print(f"\n### {label}\n$ {' '.join(command)}", flush=True)
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT)
    duration = time.perf_counter() - started
    steps.append({"label": label, "command": command, "seconds": round(duration, 3), "passed": result.returncode == 0})
    if result.returncode:
        raise RuntimeError(f"Step failed: {label} (exit {result.returncode})")


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-dvc", action="store_true", help="Use existing DVC/database state")
    parser.add_argument("--runs", type=int, default=10, help="Paired benchmark runs")
    args = parser.parse_args()
    docker = "docker" if shutil.which("docker") else "docker.exe"
    steps: list[dict] = []
    report_path = ROOT / "data" / "benchmarks" / "reproducible_demo_report.json"
    try:
        _run("Environment diagnostics", [sys.executable, "scripts/doctor.py"], steps)
        _run("Start services", [docker, "compose", "up", "-d"], steps)
        _run("Python regression suite", [sys.executable, "-m", "pytest", "-q"], steps)
        if not args.skip_dvc:
            _run("DVC pipeline", [sys.executable, "-m", "dvc", "repro"], steps)
        _run("API/data-lake smoke test", [sys.executable, "scripts/smoke_api.py"], steps)
        _run(
            "Paired ingestion benchmark",
            [sys.executable, "scripts/benchmark_ingest.py", "--runs", str(args.runs), "--batch-sizes", "1", "100"],
            steps,
        )
    finally:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        benchmark_path = ROOT / "data" / "benchmarks" / "ingest_benchmark_results.json"
        report = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "passed": bool(steps) and all(step["passed"] for step in steps),
            "steps": steps,
            "artifacts": {str(benchmark_path.relative_to(ROOT)): _sha256(benchmark_path)},
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nDemo report: {report_path}")


if __name__ == "__main__":
    main()
