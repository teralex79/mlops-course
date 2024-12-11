# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long
#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd


def read_data(filename, categorical):
    """Read data function"""

    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


def load_model(input_file: str, dv, lr, output_file: str, year, month):
    """Load model function"""

    categorical = ['PULocationID', 'DOLocationID']
    df = read_data(input_file, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    x_val = dv.transform(dicts)
    y_pred = lr.predict(x_val)

    print('predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    df_result.to_parquet(output_file, engine='pyarrow', index=False)


def main(taxi_type, year, month):
    """Main function"""

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    load_model(
        input_file=input_file,
        dv=dv,
        lr=lr,
        output_file=output_file, 
        year=year, 
        month=month
    )


if __name__ == '__main__':
    taxi_type   = str(sys.argv[1]) #'yellow'
    year        = int(sys.argv[2]) #2023
    month       = int(sys.argv[3]) #3

    main(taxi_type=taxi_type, year=year, month=month)
