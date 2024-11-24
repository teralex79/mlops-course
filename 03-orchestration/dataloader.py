import requests
from io import BytesIO
from typing import List

import pandas as pd

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader


@data_loader
def ingest_files(**kwargs) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []

    parquet_file = r'./data/yellow_tripdata_2023-03.parquet'

    df = pd.read_parquet(parquet_file)
    dfs.append(df)

    return pd.concat(dfs)