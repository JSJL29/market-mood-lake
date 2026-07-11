from src.ingestion.market_api import MarketMoodAPI


class _Response:
    text = "DATE,OPEN,HIGH,LOW,CLOSE\n01/02/2020,12,14,11,13\n01/03/2020,13,15,12,14\n"

    def raise_for_status(self):
        return None


def test_vix_history_uses_official_csv_and_start_date(monkeypatch):
    monkeypatch.setattr("src.ingestion.market_api.requests.get", lambda *args, **kwargs: _Response())
    rows = MarketMoodAPI.get_vix(period="max", start_date="2020-01-03")
    assert rows == [{"date": "2020-01-03", "vix_close": 14.0, "vix_high": 15.0, "vix_low": 12.0}]
