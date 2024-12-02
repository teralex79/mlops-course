#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip freeze | grep scikit-learn')


# In[2]:


get_ipython().system('python -V')


# In[3]:


import pickle
import pandas as pd


# In[4]:


year = 2023
month = 3
taxi_type = 'yellow'

input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
output_file = f'./output/{taxi_type}_{year:04d}-{month:02d}.parquet'


# In[5]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[6]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[7]:


df = read_data(input_file)


# In[8]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[9]:


y_pred.std()


# In[10]:


df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


# In[11]:


df


# In[12]:


df_result = pd.DataFrame()
df_result['ride_id'] = df['ride_id']
df_result['y_pred'] = y_pred


# In[13]:


df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)


# In[14]:


ls output/


# In[16]:


ls -lha ./output/


# In[ ]:




