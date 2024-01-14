import datetime
import requests
import json
import pandas as pd
import mlrun

@mlrun.handler(outputs=["parking_data_latest"])
def parkings_last_data(context):
    """
    Downloads the latest parking data from the API and returns it as a DataFrame.

    Args:
        context (mlrun.MLClientCtx): The MLRun context object.

    Returns:
        pd.DataFrame: The latest parking data as a DataFrame.
    """

    # Get the current date in the format YYYY-MM-DD
    date_str = datetime.datetime.now().strftime('%Y-%m-%d')

    # Construct the API URL with the current date and limit the results to 100 records
    API_URL = f'https://opendata.comune.bologna.it/api/explore/v2.1/catalog/datasets/disponibilita-parcheggi-storico/records?where=data%3E%3D%27{date_str}%27&order_by=data%20DESC&limit=100'

    # Define the name of the file to save the latest data
    latest_data_file = 'last_records.json'

    # Download the latest data from the API and save it to a file
    with requests.get(API_URL) as r:
        with open(latest_data_file, "wb") as f:
            f.write(r.content)

    # Read the data from the file and convert it to a DataFrame
    with open(latest_data_file) as f:
        json_data = json.load(f)
        df_latest = pd.json_normalize(json_data['results']).drop(columns=['guid', 'occupazione']).rename(columns={"coordinate.lon": "lon", "coordinate.lat": "lat"})

    return df_latest
