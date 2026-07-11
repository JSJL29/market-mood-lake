import pandas as pd

from src.transform.preprocess_to_staging import aggregate_market_proxy, clean_and_merge


def test_multi_ticker_input_is_aggregated_by_date():
    stocks = pd.DataFrame({
        "Ticker": ["A", "A", "B", "B"],
        "Date": ["2026-01-01", "2026-01-02", "2026-01-01", "2026-01-02"],
        "Close": [100.0, 110.0, 200.0, 180.0],
        "Volume": [10, 11, 20, 21],
    })
    proxy = aggregate_market_proxy(stocks)
    assert len(proxy) == 1
    assert proxy.iloc[0]["constituent_count"] == 2
    assert proxy.iloc[0]["volume"] == 32
    # 100 * exp(mean(log(1.1), log(0.9)))
    assert abs(proxy.iloc[0]["close"] - 99.49874371) < 1e-6


def test_sentiment_alignment_uses_past_values_only():
    dates = pd.bdate_range("2026-01-01", periods=80)
    prices = pd.DataFrame({
        "date": dates,
        "close": [100.0 + i for i in range(len(dates))],
        "volume": [1000] * len(dates),
    })
    vix = pd.DataFrame({
        "date": ["2026-03-19", "2026-03-23"],
        "vix_close": [20.0, 40.0],
    })
    fear = pd.DataFrame({
        "date": ["2026-03-19"],
        "fear_greed_score": [55.0],
    })

    merged = clean_and_merge(prices, vix, fear)
    march_20 = merged.loc[merged["date"] == "2026-03-20"].iloc[0]
    assert march_20["vix_close"] == 20.0
    assert march_20["fear_greed_score"] == 55.0


def test_sentiment_is_not_propagated_beyond_tolerance():
    dates = pd.bdate_range("2026-01-01", periods=80)
    prices = pd.DataFrame({"date": dates, "close": range(100, 180)})
    mood = pd.DataFrame({"date": ["2026-01-01"], "vix_close": [20.0]})
    merged = clean_and_merge(prices, mood, pd.DataFrame(columns=["date"]))
    assert pd.isna(merged.iloc[-1]["vix_close"])
