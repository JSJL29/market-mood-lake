import pandas as pd
import pytest

from src.transform.process_to_curated import FEATURE_COLUMNS, build_sequences, insert_to_mongodb


def test_build_sequences_rejects_missing_columns():
    with pytest.raises(ValueError, match="Colonnes absentes"):
        build_sequences(pd.DataFrame({"date": ["2026-01-01"]}))


def test_build_sequences_returns_expected_window_count():
    rows = 5
    data = {column: [float(i + 1) for i in range(rows)] for column in FEATURE_COLUMNS}
    data.update({"date": pd.date_range("2026-01-01", periods=rows), "target_up": [0, 1, 0, 1, 0]})
    sequences = build_sequences(pd.DataFrame(data), window_size=2)
    assert len(sequences) == rows - 2 + 1
    assert len(sequences[0]["features"]) == 2
    assert sequences[0]["window_end_date"].startswith("2026-01-02")
    assert sequences[0]["label"] == 1


def test_empty_mongodb_insert_is_rejected_before_connecting():
    with pytest.raises(ValueError, match="Aucun document"):
        insert_to_mongodb([])


def test_windows_do_not_cross_large_calendar_gaps():
    data = {column: [1.0, 2.0, 3.0, 4.0] for column in FEATURE_COLUMNS}
    data.update({
        "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-02-01", "2026-02-02"]),
        "target_up": [0, 1, 0, 1],
    })
    sequences = build_sequences(pd.DataFrame(data), window_size=2)
    assert [sequence["window_end_date"][:10] for sequence in sequences] == ["2026-01-02", "2026-02-02"]
