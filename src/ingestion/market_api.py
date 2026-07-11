import io
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone

import requests
import pandas as pd
import yfinance as yf
import boto3


class MarketMoodAPI:
    """Récupère le VIX (via yfinance) et le Fear & Greed Index."""

    # CNN ne publie pas de dataset historique officiel et bloque le scraping
    # direct (User-Agent strict, erreurs 418/500 sur les requêtes larges).
    # On utilise à la place une copie historique maintenue quotidiennement
    # sur GitHub (2011-aujourd'hui), qui combine plusieurs sources fiables
    # dont le flux CNN lui-même pour la période récente.
    FEAR_GREED_HISTORY_URL = "https://raw.githubusercontent.com/whit3rabbit/fear-greed-data/main/fear-greed.csv"
    FEAR_GREED_CNN_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    VIX_HISTORY_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"

    @staticmethod
    def get_vix(period="5d", start_date=None):
        """Récupère le VIX, avec l'historique officiel Cboe en priorité."""
        if start_date or period == "max":
            response = requests.get(MarketMoodAPI.VIX_HISTORY_URL, timeout=30)
            response.raise_for_status()
            cboe = pd.read_csv(io.StringIO(response.text))
            cboe.columns = [column.strip().lower() for column in cboe.columns]
            cboe["date"] = pd.to_datetime(cboe["date"], format="%m/%d/%Y", errors="coerce")
            cboe = cboe.dropna(subset=["date", "close"])
            if start_date:
                cboe = cboe[cboe["date"] >= pd.Timestamp(start_date)]
            return [
                {
                    "date": row["date"].strftime("%Y-%m-%d"),
                    "vix_close": float(row["close"]),
                    "vix_high": float(row["high"]),
                    "vix_low": float(row["low"]),
                }
                for _, row in cboe.iterrows()
            ]

        vix = yf.Ticker("^VIX")
        hist = vix.history(period=period)

        records = []
        for date, row in hist.iterrows():
            records.append({
                "date": date.strftime("%Y-%m-%d"),
                "vix_close": float(row["Close"]),
                "vix_high": float(row["High"]),
                "vix_low": float(row["Low"]),
            })
        return records

    @staticmethod
    def get_fear_greed(start_date=None):
        """
        Récupère l'historique du Fear & Greed Index depuis le CSV
        maintenu sur GitHub (mise à jour quotidienne, couvre 2011 à
        aujourd'hui). Beaucoup plus fiable que le scraping direct de
        l'endpoint CNN pour un usage reproductible.

        Parameters
        ----------
        start_date : str, optional
            Ne garde que les lignes à partir de cette date (YYYY-MM-DD).
        """
        response = requests.get(MarketMoodAPI.FEAR_GREED_HISTORY_URL, timeout=15)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))

        if start_date:
            df = df[df["Date"] >= start_date]

        records = [
            {"date": row["Date"], "fear_greed_score": float(row["Fear Greed"])}
            for _, row in df.iterrows()
        ]
        return records

    @staticmethod
    def get_fear_greed_from_cnn_live(start_date=None):
        """
        Alternative : scrape l'endpoint live CNN directement. Conservée
        pour référence, mais moins fiable (headers stricts, erreurs 500
        sur de grandes plages de dates) que get_fear_greed().
        """
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
        }
        url = MarketMoodAPI.FEAR_GREED_CNN_URL
        if start_date:
            url = f"{url}/{start_date}"

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        records = []
        for point in data.get("fear_and_greed_historical", {}).get("data", []):
            ts = point.get("x", 0) / 1000
            records.append({
                "date": datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d"),
                "fear_greed_score": float(point.get("y", 0)),
            })
        return records


def upload_to_s3(payload, bucket_name, key, endpoint_url):
    """Upload un payload JSON vers S3, comme dans hn_api.py (TP6)."""
    s3_client = boto3.client('s3', endpoint_url=endpoint_url)
    body = json.dumps(payload).encode('utf-8')

    try:
        s3_client.put_object(Bucket=bucket_name, Key=key, Body=body)
        print(f"Données téléversées avec succès dans s3://{bucket_name}/{key}")
    except Exception as e:
        print(f"Erreur lors du téléversement : {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Fetch VIX and Fear & Greed data into the raw bucket')
    parser.add_argument('--period', type=str, default='5d', help='Période yfinance pour le VIX (ex: 5d, 1mo, max)')
    parser.add_argument('--start_date', type=str, default=None,
                         help="Date de départ (YYYY-MM-DD) pour l'historique Fear & Greed. Sans cette option, tout l'historique disponible (depuis 2011) est récupéré.")
    parser.add_argument('--bucket_name', type=str, default='raw', help='Nom du bucket S3 raw')
    parser.add_argument('--endpoint-url', type=str, default='http://localhost:4566',
                         help='URL du endpoint S3 (LocalStack)')
    parser.add_argument('--output-path', type=Path, default=None,
                         help='Snapshot JSON local suivi par DVC')
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    print("Récupération du VIX...")
    vix_records = MarketMoodAPI.get_vix(args.period, args.start_date)
    if not vix_records:
        raise RuntimeError("Aucune donnée VIX récupérée")

    print("Récupération du Fear & Greed Index...")
    try:
        fg_records = MarketMoodAPI.get_fear_greed(args.start_date)
    except Exception as e:
        print(f"Fear & Greed indisponible ({e}), poursuite avec le VIX seul.")
        fg_records = []

    if not fg_records:
        raise RuntimeError("Aucune donnée Fear & Greed récupérée")

    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "vix": vix_records,
        "fear_greed": fg_records,
    }

    if args.output_path:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        print(f"Snapshot market mood écrit dans {args.output_path}")

    upload_to_s3(payload, args.bucket_name, f"market_mood_{timestamp}.json", args.endpoint_url)


if __name__ == "__main__":
    main()
