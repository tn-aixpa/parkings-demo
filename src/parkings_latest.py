from digitalhub_runtime_python import handler
import datetime
import json
import os
import pandas as pd
import requests
from sqlalchemy import create_engine

def startOfDay():
    today = datetime.datetime.now()
    # Create a datetime object for the start of today 
    start_of_today = today.replace(hour=0, minute=0, second=0, microsecond=0)
    return start_of_today
    
@handler()
def parkings_last_data():
    date_str = datetime.datetime.now().strftime('%Y-%m-%d')
    latest_data_file = 'latest_records.json'
    API_URL = f'https://opendata.comune.bologna.it/api/explore/v2.1/catalog/datasets/disponibilita-parcheggi-storico/records?where=data%3E%3D%27{date_str}%27&order_by=data%20DESC&limit=100'
    
    # Download the latest data from the API and save it to a file
    with requests.get(API_URL) as r:
        with open(latest_data_file, "wb") as f:
            f.write(r.content)
            
        # Read the data from the file and convert it to a DataFrame
        with open(latest_data_file) as f:
            json_data = json.load(f)
            df_latest = pd.json_normalize(json_data['results']).drop(columns=['guid', 'occupazione']).rename(columns={"coordinate.lon": "lon", "coordinate.lat": "lat"})

    # convert 'data' column to datetime
    df_latest.data = df_latest.data.astype('datetime64[ns, UTC]')                     
    # write data to database
    USERNAME = os.getenv("POSTGRES_USER")
    PASSWORD = os.getenv("POSTGRES_PASSWORD")
    engine = create_engine('postgresql+psycopg2://'+USERNAME+':'+PASSWORD+'@database-postgres-cluster/digitalhub')
    
    with engine.connect() as connection: 
        try: connection.execute("DELETE FROM parkings_latest where data >= '" + str(parkings_last_data().min().data) + "' and data < '" + date_str + "'")
        except: pass

    start_of_today = startOfDay()
    with engine.connect() as connection: 
        try: connection.execute("DELETE FROM parkings_latest WHERE data < " + start_of_today)
        except: pass            

    df_latest.to_sql('parkings_latest', engine, if_exists="append")   
