from scripts.acquire_historical_data import acquire


def test_existing_historical_file_is_reused(tmp_path):
    destination = tmp_path / "history.csv"
    destination.write_text("ticker,date,close\nSPY,2026-01-01,100\n", encoding="utf-8")
    assert acquire(destination) == "existing"


def test_missing_source_has_actionable_error(tmp_path):
    try:
        acquire(tmp_path / "missing.csv")
    except FileNotFoundError as exc:
        assert "HISTORICAL_DATA_URL" in str(exc)
    else:
        raise AssertionError("missing source should fail")
