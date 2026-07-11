import os
import re
import tempfile
import argparse
import pandas as pd
import boto3


def unpack_stocks_data(input_dir, bucket_name, output_file_name, endpoint_url="http://localhost:4566",
                        max_tickers=None, input_file=None):
    """
    Ingestion d'un dataset multi-actions (cross-sectional) vers le bucket raw.

    Gère deux formats Kaggle courants :
    - Un seul CSV combiné avec une colonne 'Ticker'/'Symbol'.
    - Plusieurs CSV, un par action (ex: AAPL.csv, MSFT.csv), sans colonne
      ticker : le ticker est alors déduit du nom de fichier.

    Parameters
    ----------
    input_dir : str
        Répertoire contenant le(s) fichier(s) CSV.
    bucket_name : str
        Nom du bucket S3 raw.
    output_file_name : str
        Nom du fichier combiné uploadé sur S3.
    endpoint_url : str
        URL du endpoint S3 (LocalStack).
    max_tickers : int, optional
        Limite le nombre de tickers conservés (utile pour garder un temps
        de traitement raisonnable sur une machine de développement).
    """
    s3 = boto3.client('s3', endpoint_url=endpoint_url)
    data_frames = []
    if input_file:
        input_path = os.path.abspath(input_file)
        input_dir = os.path.dirname(input_path)
        csv_files = [os.path.basename(input_path)]
    else:
        csv_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".csv"))

    if not csv_files:
        raise FileNotFoundError(f"Aucun fichier CSV trouvé dans {input_dir}")

    for file_name in csv_files:
        file_path = os.path.join(input_dir, file_name)
        print(f"Lecture de {file_path}")
        data = pd.read_csv(file_path)
        data.columns = [c.strip().lower().replace(" ", "_") for c in data.columns]

        # Normalise le nom de la colonne ticker si elle existe déjà
        for candidate in ["ticker", "symbol", "symbols"]:
            if candidate in data.columns:
                data = data.rename(columns={candidate: "ticker"})
                break

        # Sinon, déduit le ticker à partir du nom du fichier (ex: AAPL.csv -> AAPL)
        if "ticker" not in data.columns:
            ticker = re.sub(r"\.csv$", "", file_name, flags=re.IGNORECASE).upper()
            data["ticker"] = ticker

        data_frames.append(data)

    combined_data = pd.concat(data_frames, ignore_index=True)
    required = {"date", "close", "ticker"}
    missing = required - set(combined_data.columns)
    if missing:
        raise ValueError(f"CSV historique incomplet, colonnes manquantes: {sorted(missing)}")
    if combined_data.empty:
        raise ValueError("CSV historique vide")
    print(f"Tous les fichiers combinés : {len(combined_data)} lignes, "
          f"{combined_data['ticker'].nunique()} tickers.")

    if max_tickers is not None:
        kept_tickers = sorted(combined_data["ticker"].unique())[:max_tickers]
        combined_data = combined_data[combined_data["ticker"].isin(kept_tickers)]
        print(f"Limité à {max_tickers} tickers : {len(combined_data)} lignes conservées.")

    combined_csv_path = os.path.join(tempfile.gettempdir(), output_file_name)
    combined_data.to_csv(combined_csv_path, index=False)
    print(f"Fichier combiné sauvegardé localement : {combined_csv_path}.")

    s3.upload_file(combined_csv_path, bucket_name, output_file_name)
    print(f"Fichier téléversé dans le bucket '{bucket_name}' sous le nom '{output_file_name}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingestion multi-actions (cross-sectional) vers le bucket raw")
    parser.add_argument("--input_dir", type=str, required=True, help="Répertoire contenant le(s) CSV multi-actions")
    parser.add_argument("--input_file", type=str, default=None, help="CSV source explicite (recommandé pour DVC/Airflow)")
    parser.add_argument("--bucket_name", type=str, default="raw", help="Nom du bucket S3")
    parser.add_argument("--output_file_name", type=str, default="stocks_combined.csv",
                         help="Nom du fichier de sortie sur S3")
    parser.add_argument("--endpoint-url", type=str, default="http://localhost:4566",
                         help="URL du endpoint S3 (LocalStack)")
    parser.add_argument("--max_tickers", type=int, default=None,
                         help="Limite le nombre de tickers conservés (ex: 50 pour un traitement rapide)")
    args = parser.parse_args()

    unpack_stocks_data(
        args.input_dir, args.bucket_name, args.output_file_name, args.endpoint_url,
        args.max_tickers, args.input_file,
    )
