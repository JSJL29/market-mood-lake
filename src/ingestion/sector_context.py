"""
Ingestion d'une nouvelle source de donnees sectorielle pour le pipeline
cross-sectional de Market Mood Lake.

Ce script cree deux fichiers dans le bucket raw S3/LocalStack :
- sector_etfs.csv : historiques quotidiens des ETF sectoriels SPDR.
- ticker_sector_map.csv : mapping ticker -> secteur -> ETF sectoriel.

L'objectif est d'ajouter au modele un contexte relatif :
une action monte-t-elle parce que tout son secteur monte, ou parce qu'elle
surperforme reellement son secteur ?

Exemple :
    python -m src.ingestion.sector_context \
        --bucket_name raw \
        --stocks_file stocks_combined.csv \
        --endpoint-url http://localhost:4566 \
        --start_date 2015-01-01
"""

from __future__ import annotations

import argparse
import io
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Optional

import boto3
import pandas as pd
import yfinance as yf


SECTOR_TO_ETF: Dict[str, str] = {
    # Noms yfinance frequents
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Communication Services": "XLC",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    # Noms GICS / variantes courantes
    "Information Technology": "XLK",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Communication Services": "XLC",
    "Materials": "XLB",
}

DEFAULT_ETFS = sorted(set(SECTOR_TO_ETF.values()) | {"SPY"})


def _s3_client(endpoint_url: str):
    return boto3.client("s3", endpoint_url=endpoint_url)


def _read_raw_csv(endpoint_url: str, bucket: str, key: str) -> pd.DataFrame:
    s3 = _s3_client(endpoint_url)
    response = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(response["Body"].read()))


def _upload_dataframe_to_s3(
    df: pd.DataFrame,
    *,
    endpoint_url: str,
    bucket: str,
    key: str,
) -> None:
    tmp_path = Path(tempfile.gettempdir()) / key
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp_path, index=False)
    _s3_client(endpoint_url).upload_file(str(tmp_path), bucket, key)
    print(f"Upload raw terminé : s3://{bucket}/{key} ({len(df)} lignes)")


def _normalise_stock_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    for candidate in ["ticker", "symbol", "symbols"]:
        if candidate in df.columns:
            df = df.rename(columns={candidate: "ticker"})
            break
    if "ticker" not in df.columns:
        raise ValueError("stocks_combined.csv doit contenir une colonne ticker/symbol.")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    return df


def _map_sector_to_etf(sector: Optional[str]) -> str:
    if sector is None or pd.isna(sector):
        return "SPY"
    sector = str(sector).strip()
    return SECTOR_TO_ETF.get(sector, "SPY")


def download_sector_etfs(
    etfs: Iterable[str],
    *,
    start_date: str,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Télécharge les prix quotidiens des ETF sectoriels via yfinance."""
    etfs = sorted(set(etfs))
    print(f"Téléchargement yfinance des ETF sectoriels : {', '.join(etfs)}")
    data = yf.download(
        tickers=etfs,
        start=start_date,
        end=end_date,
        progress=False,
        group_by="ticker",
        auto_adjust=False,
        threads=True,
    )

    rows = []
    if isinstance(data.columns, pd.MultiIndex):
        for etf in etfs:
            if etf not in data.columns.get_level_values(0):
                print(f"  ETF absent du retour yfinance : {etf}")
                continue
            sub = data[etf].reset_index()
            close_col = "Close" if "Close" in sub.columns else "Adj Close"
            volume_col = "Volume" if "Volume" in sub.columns else None
            for _, row in sub.iterrows():
                if pd.isna(row.get(close_col)):
                    continue
                rows.append(
                    {
                        "date": pd.to_datetime(row["Date"]).strftime("%Y-%m-%d"),
                        "sector_etf": etf,
                        "sector_close": float(row[close_col]),
                        "sector_volume": int(row[volume_col]) if volume_col and pd.notna(row.get(volume_col)) else 0,
                    }
                )
    else:
        # Cas d'un seul ticker, gardé par sécurité.
        sub = data.reset_index()
        close_col = "Close" if "Close" in sub.columns else "Adj Close"
        volume_col = "Volume" if "Volume" in sub.columns else None
        etf = etfs[0]
        for _, row in sub.iterrows():
            if pd.isna(row.get(close_col)):
                continue
            rows.append(
                {
                    "date": pd.to_datetime(row["Date"]).strftime("%Y-%m-%d"),
                    "sector_etf": etf,
                    "sector_close": float(row[close_col]),
                    "sector_volume": int(row[volume_col]) if volume_col and pd.notna(row.get(volume_col)) else 0,
                }
            )

    result = pd.DataFrame(rows).drop_duplicates(subset=["date", "sector_etf"])
    result = result.sort_values(["sector_etf", "date"]).reset_index(drop=True)
    if result.empty:
        raise RuntimeError("Aucune donnée ETF sectorielle téléchargée.")
    return result


def build_ticker_sector_map(
    stocks_df: pd.DataFrame,
    *,
    local_sector_map: Optional[str] = None,
    skip_yfinance_lookup: bool = False,
) -> pd.DataFrame:
    """Construit ticker -> sector -> sector_etf.

    Priorités :
    1. fichier CSV fourni par --local_sector_map ;
    2. colonne sector/gics_sector déjà présente dans stocks_combined.csv ;
    3. lookup yfinance ticker.info['sector'] ;
    4. fallback SPY si secteur introuvable.
    """
    stocks_df = _normalise_stock_columns(stocks_df)
    tickers = sorted(stocks_df["ticker"].dropna().unique().tolist())

    if local_sector_map:
        sector_map = pd.read_csv(local_sector_map)
        sector_map.columns = [c.strip().lower().replace(" ", "_") for c in sector_map.columns]
        if "ticker" not in sector_map.columns:
            raise ValueError("Le fichier local_sector_map doit contenir une colonne ticker.")
        if "sector" not in sector_map.columns:
            for candidate in ["gics_sector", "yfinance_sector"]:
                if candidate in sector_map.columns:
                    sector_map = sector_map.rename(columns={candidate: "sector"})
                    break
        if "sector" not in sector_map.columns:
            raise ValueError("Le fichier local_sector_map doit contenir sector ou gics_sector.")
        sector_map = sector_map[["ticker", "sector"]].copy()
        sector_map["ticker"] = sector_map["ticker"].astype(str).str.upper().str.strip()
        sector_map["sector_etf"] = sector_map["sector"].map(_map_sector_to_etf)
        return sector_map.drop_duplicates(subset=["ticker"]).reset_index(drop=True)

    for candidate in ["sector", "gics_sector", "yfinance_sector"]:
        if candidate in stocks_df.columns:
            print(f"Mapping secteur trouvé directement dans stocks_combined.csv via la colonne {candidate!r}.")
            sector_map = stocks_df[["ticker", candidate]].drop_duplicates(subset=["ticker"]).copy()
            sector_map = sector_map.rename(columns={candidate: "sector"})
            sector_map["sector_etf"] = sector_map["sector"].map(_map_sector_to_etf)
            return sector_map.reset_index(drop=True)

    if skip_yfinance_lookup:
        print("Aucun secteur trouvé ; fallback : tous les tickers utilisent SPY.")
        return pd.DataFrame(
            {
                "ticker": tickers,
                "sector": ["UNKNOWN"] * len(tickers),
                "sector_etf": ["SPY"] * len(tickers),
            }
        )

    print(f"Lookup yfinance des secteurs pour {len(tickers)} tickers. Ça peut prendre quelques minutes.")
    rows = []
    for i, ticker in enumerate(tickers, start=1):
        sector = None
        try:
            info = yf.Ticker(ticker).get_info()
            sector = info.get("sector")
        except Exception as exc:  # yfinance peut échouer ticker par ticker
            print(f"  Warning secteur introuvable pour {ticker}: {exc}")
        rows.append(
            {
                "ticker": ticker,
                "sector": sector if sector else "UNKNOWN",
                "sector_etf": _map_sector_to_etf(sector),
            }
        )
        if i % 25 == 0 or i == len(tickers):
            print(f"  {i}/{len(tickers)} tickers traités")

    return pd.DataFrame(rows)


def ingest_sector_context(
    *,
    bucket_name: str,
    stocks_file: str,
    endpoint_url: str,
    start_date: str,
    end_date: Optional[str],
    sector_etfs_file: str,
    sector_map_file: str,
    local_sector_map: Optional[str],
    skip_yfinance_lookup: bool,
) -> None:
    print("Lecture du fichier multi-actions depuis raw...")
    stocks_df = _read_raw_csv(endpoint_url, bucket_name, stocks_file)

    sector_map = build_ticker_sector_map(
        stocks_df,
        local_sector_map=local_sector_map,
        skip_yfinance_lookup=skip_yfinance_lookup,
    )
    etfs = sorted(set(sector_map["sector_etf"].dropna().tolist()) | {"SPY"})
    sector_etfs = download_sector_etfs(etfs, start_date=start_date, end_date=end_date)

    _upload_dataframe_to_s3(
        sector_map,
        endpoint_url=endpoint_url,
        bucket=bucket_name,
        key=sector_map_file,
    )
    _upload_dataframe_to_s3(
        sector_etfs,
        endpoint_url=endpoint_url,
        bucket=bucket_name,
        key=sector_etfs_file,
    )

    print("\nRésumé mapping secteur :")
    print(sector_map["sector_etf"].value_counts(dropna=False).sort_index())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingestion contexte sectoriel vers raw S3")
    parser.add_argument("--bucket_name", type=str, default="raw")
    parser.add_argument("--stocks_file", type=str, default="stocks_combined.csv")
    parser.add_argument("--endpoint-url", type=str, default="http://localhost:4566")
    parser.add_argument("--start_date", type=str, default="2015-01-01")
    parser.add_argument("--end_date", type=str, default=None)
    parser.add_argument("--sector_etfs_file", type=str, default="sector_etfs.csv")
    parser.add_argument("--sector_map_file", type=str, default="ticker_sector_map.csv")
    parser.add_argument(
        "--local_sector_map",
        type=str,
        default=None,
        help="CSV optionnel avec colonnes ticker,sector pour éviter le lookup yfinance ticker par ticker.",
    )
    parser.add_argument(
        "--skip_yfinance_lookup",
        action="store_true",
        help="Si aucun secteur n'est disponible, fallback SPY pour tous les tickers au lieu d'appeler yfinance.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ingest_sector_context(
        bucket_name=args.bucket_name,
        stocks_file=args.stocks_file,
        endpoint_url=args.endpoint_url,
        start_date=args.start_date,
        end_date=args.end_date,
        sector_etfs_file=args.sector_etfs_file,
        sector_map_file=args.sector_map_file,
        local_sector_map=args.local_sector_map,
        skip_yfinance_lookup=args.skip_yfinance_lookup,
    )
