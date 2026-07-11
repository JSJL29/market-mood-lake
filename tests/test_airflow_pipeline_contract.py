from pathlib import Path


def test_main_airflow_dag_contains_both_data_branches():
    source = Path("dags/market_mood_pipeline.py").read_text(encoding="utf-8")
    assert 'task_id="preprocess_to_staging"' in source
    assert 'task_id="process_to_curated"' in source
    assert 'task_id="preprocess_to_staging_xs"' in source
    assert 'task_id="process_to_curated_xs"' in source
    assert "fetch_market_mood >> preprocess_to_staging_xs >> process_to_curated_xs" in source
