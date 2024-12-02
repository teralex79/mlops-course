import pickle
import pandas as pd
import sys
import os

categorical = ['PULocationID', 'DOLocationID']
taxi_type   = sys.argv[1] #'yellow'
year        = int(sys.argv[2]) #2023
month       = int(sys.argv[3]) #3   

def read_data(filename: str):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def load_model(input_file: str, dv, model, output_file: str):
    df = read_data(input_file)
    
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['y_pred'] = y_pred

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

def run():            
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'./output/{taxi_type}_{year:04d}-{month:02d}.parquet'

    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    
    load_model(
        input_file=input_file,
        dv=dv,
        model=model,
        output_file=output_file
    )


if __name__ == '__main__':
    run()