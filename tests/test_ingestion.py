import numpy as np

from src.api import routes_ingest


def test_naive_and_vectorized_features_match():
    ticks = [
        routes_ingest.MarketTick(date="2026-01-01", close=100, vix_close=10),
        routes_ingest.MarketTick(date="2026-01-02", close=110, vix_close=20),
    ]
    naive_returns, naive_vix = routes_ingest._naive_features(ticks)
    fast_returns, fast_vix = routes_ingest._vectorized_features(
        np.array([100.0, 110.0]), np.array([10.0, 20.0])
    )
    np.testing.assert_allclose(fast_returns, naive_returns)
    np.testing.assert_allclose(fast_vix, naive_vix)


def test_fast_single_item_skips_jit(monkeypatch):
    inserted = {}
    monkeypatch.setattr(routes_ingest, "_vectorized_features", lambda *_: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(routes_ingest, "_insert_batch", lambda ticks, returns, vix, **kwargs: inserted.update(
        ticks=ticks, returns=returns, vix=vix
    ))
    payload = routes_ingest.IngestPayload(data=[{"date": "2026-01-01", "close": 100.0}])
    import asyncio
    result = asyncio.run(routes_ingest.ingest_fast(payload))
    assert result["batch_size"] == 1
    np.testing.assert_array_equal(inserted["returns"], [0.0])
    np.testing.assert_array_equal(inserted["vix"], [0.0])
