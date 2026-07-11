from scripts.acquire_historical_data import acquire


def test_existing_historical_file_is_reused(tmp_path):
    destination = tmp_path / "history.csv"
    destination.write_text("ticker,date,close\nSPY,2026-01-01,100\n", encoding="utf-8")
    assert acquire(destination) == "existing"


def test_missing_source_generates_offline_demo(tmp_path):
    destination = tmp_path / "historical.csv"
    assert acquire(destination) == "generated-demo"
    assert destination.stat().st_size > 0
    assert destination.read_text(encoding="utf-8").splitlines()[0].startswith("ticker,date,")


def test_missing_source_has_actionable_error(tmp_path):
    try:
        acquire(tmp_path / "missing.csv", allow_demo_fallback=False)
    except FileNotFoundError as exc:
        assert "HISTORICAL_DATA_URL" in str(exc)
    else:
        raise AssertionError("missing source should fail")
