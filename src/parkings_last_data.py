
import mlrun
import pandas as pd
import requests
import json

@mlrun.handler(outputs=["parking_data_latest"])
def parkings_last_data(context):
    API_URL = 'https://opendata.comune.bologna.it/api/explore/v2.1/catalog/datasets/disponibilita-parcheggi-storico/records?where=data%20%3E%20%272023-12-20%27&limit=100'
    latest_data_file = 'last_records.json'
    with requests.get(API_URL) as r:
        with open(latest_data_file, "wb") as f:
            f.write(r.content)
    with open(latest_data_file) as f:
        json_data = json.load(f)
        df_latest = pd.json_normalize(json_data['results']).drop(columns=['guid', 'occupazione']).rename(columns={"coordinate.lon": "lon", "coordinate.lat": "lat"})
    return df_latest
