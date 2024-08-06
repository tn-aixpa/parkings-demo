
import pandas as pd
import streamlit as st

import datetime 
import requests
import json

date_str = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
parking_str='Riva Reno'
API_URL = f'https://opendata.comune.bologna.it/api/explore/v2.1/catalog/datasets/disponibilita-parcheggi-storico/records?where=data%3C%3D%27{date_str}%27%20and%20parcheggio%3D%27{parking_str}%27&order_by=data%20DESC&limit=100'

SERVICE_URL = 's-python-python-serve-a101972d-4860-4ae5-b6c0-83da27fe5bda.digitalhub-tenant1.svc.cluster.local:8080'


latest_data_file = 'last_records.json'

with requests.get(API_URL) as r:
    with open(latest_data_file, "wb") as f:
        f.write(r.content)

with open(latest_data_file) as f:
    json_data = json.load(f)
    df_latest = pd.json_normalize(json_data['results']).drop(columns=['guid', 'occupazione']).rename(columns={"coordinate.lon": "lon", "coordinate.lat": "lat"})
    df_latest.data = df_latest.data.astype('datetime64')
    df_latest['value'] = df_latest.posti_occupati / df_latest.posti_totali
    df_latest['date'] = df_latest.data.dt.round('30min')
    df_latest = df_latest.drop(columns=['parcheggio'])
    df_latest = df_latest.groupby('date').agg({'value': 'mean'})


jsonstr = df_latest.reset_index().to_json(orient='records')
arr = json.loads(jsonstr)

with requests.post(f'http://{SERVICE_URL}', json={"inference_input":arr}) as r:
    res = r.json()
print(res)

rdf = pd.read_json(json.dumps(res), orient="records")

st.write("""prediction""")
st.line_chart(rdf, x="date", y="value")
