from scripts import benchmark_ingest


class _Health:
    def raise_for_status(self):
        return None


def test_benchmark_is_paired_balanced_and_uses_fresh_dates(monkeypatch):
    calls = []

    def fake_call(base_url, endpoint, payload, timeout):
        calls.append((endpoint, payload["data"][0]["date"]))
        server = 0.010 if endpoint == "/ingest" else 0.004
        return {"wall_seconds": server + 0.001, "server_seconds": server}

    monkeypatch.setattr(benchmark_ingest.requests, "get", lambda *args, **kwargs: _Health())
    reset = _Health()
    reset.json = lambda: {"deleted_rows": 0}
    monkeypatch.setattr(benchmark_ingest.requests, "delete", lambda *args, **kwargs: reset)
    monkeypatch.setattr(benchmark_ingest, "call_endpoint", fake_call)
    result = benchmark_ingest.benchmark("http://api", [1], runs=4, timeout=1)
    pairs = result["results"]["1"]["pairs"]

    assert pairs[0]["order"] == ["/ingest", "/ingest_fast"]
    assert pairs[1]["order"] == ["/ingest_fast", "/ingest"]
    measured_dates = [date for _, date in calls[2:]]  # exclude warm-up
    assert len(measured_dates) == len(set(measured_dates))
    assert result["results"]["1"]["paired_improvement"]["median_percent"] == 60.0
    assert result["all_requirements_pass"] is True


def test_benchmark_rejects_too_few_pairs():
    try:
        benchmark_ingest.benchmark("http://api", [1], runs=3, timeout=1)
    except ValueError as exc:
        assert "4 paired runs" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
