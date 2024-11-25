from typing import List

import pandas as pd

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader


@data_loader
def ingest_files(**kwargs) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []

#    for i in range(1, 4):
       
    parquet_file = f'./data/yellow_tripdata_2023-03.parquet'
    print(parquet_file)

    df = pd.read_parquet(parquet_file) 
    dfs.append(df)

    return pd.concat(dfs)