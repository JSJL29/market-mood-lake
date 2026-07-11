"""Fast, read-only environment diagnostics for evaluators."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def _compose_works(command: str) -> bool:
    try:
        result = subprocess.run(
            [command, "compose", "version"], capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    checks = []
    checks.append((sys.version_info >= (3, 10), f"Python {sys.version.split()[0]} (>=3.10)"))
    docker_command = "docker" if shutil.which("docker") else None
    if docker_command is None and shutil.which("docker.exe"):
        docker_command = "docker.exe"
    checks.append((docker_command is not None, "docker executable"))
    compose_ready = False
    if docker_command:
        compose_ready = _compose_works(docker_command)
        if not compose_ready and docker_command == "docker" and shutil.which("docker.exe"):
            compose_ready = _compose_works("docker.exe")
    checks.append((compose_ready, "docker compose plugin (native or Docker Desktop)"))
    if compose_ready and docker_command:
        compose_executable = docker_command
        if not _compose_works(compose_executable) and shutil.which("docker.exe"):
            compose_executable = "docker.exe"
        try:
            compose_config = subprocess.run(
                [compose_executable, "compose", "config", "--quiet"],
                cwd=repository, capture_output=True, text=True, timeout=30,
            )
            checks.append((compose_config.returncode == 0, "docker-compose.yml validation"))
        except (OSError, subprocess.TimeoutExpired):
            checks.append((False, "docker-compose.yml validation"))
    free_gib = shutil.disk_usage(repository).free / (1024 ** 3)
    # An already-built stack needs little temporary space; a first Docker build
    # should still be started with roughly 10 GiB free (documented in README).
    checks.append((free_gib >= 0.5, f"free disk space {free_gib:.1f} GiB (>=0.5 GiB runtime)"))
    for required in ("dvc.yaml", "params.yaml", "docker-compose.yml"):
        checks.append(((repository / required).is_file(), f"project file {required}"))
    try:
        import dvc  # noqa: F401
        checks.append((True, "DVC Python package"))
    except ImportError:
        checks.append((False, "DVC Python package"))
    try:
        import pytest  # noqa: F401
        checks.append((True, "pytest Python package"))
    except ImportError:
        checks.append((False, "pytest Python package"))
    for passed, label in checks:
        print(f"[{'OK' if passed else 'FAIL'}] {label}")
    failed = sum(not passed for passed, _ in checks)
    print(f"Environment: {'ready' if failed == 0 else f'{failed} problem(s)'}")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
