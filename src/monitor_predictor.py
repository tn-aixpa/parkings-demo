import mlrun
import datetime 
import requests
import json
import pandas as pd

@mlrun.handler()
def monitor_predictor(context):
    """
    Monitor and predict parking occupancy.

    Args:
    context (MLRunContext): the runtime context object
    """

    # get project object
    project = context.get_project_object()

    # get serving predictor function
    serving_fn = project.get_function('serving-predictor')

    # get current date and time as string
    date_str = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

    # get parkings dataset and convert it to a dataframe
    parkings_df = mlrun.get_dataitem('store://datasets/parcheggi/extract-parkings_parkings').as_df()

    # initialize an empty dataframe for predictions
    pred_df = pd.DataFrame(columns=['parcheggio', 'datetime', 'predicted_mean'])

    # iterate over each parking in the dataset
    for index, row in parkings_df.iterrows():
        parking_str = row['parcheggio']

        # construct API URL based on parking and current date
        API_URL = f'https://opendata.comune.bologna.it/api/explore/v2.1/catalog/datasets/disponibilita-parcheggi-storico/records?where=data%3C%3D%27{date_str}%27%20and%20parcheggio%3D%27{parking_str}%27&order_by=data%20DESC&limit=100'

        # define the file to store the latest data
        latest_data_file = 'last_records.json'

        # fetch data from the API and save it to a file
        with requests.get(API_URL) as r:
            with open(latest_data_file, "wb") as f:
                f.write(r.content)

        # read the latest data from the file and process it
        with open(latest_data_file) as f:
            json_data = json.load(f)
            df_latest = pd.json_normalize(json_data['results']).drop(columns=['guid', 'occupazione']).rename(columns={"coordinate.lon": "lon", "coordinate.lat": "lat"})
            df_latest.data = df_latest.data.astype('datetime64[ms, UTC]')
            df_latest['value'] = df_latest.posti_occupati / df_latest.posti_totali
            df_latest['date'] = df_latest.data.dt.round('30min')
            df_latest = df_latest.drop(columns=['parcheggio'])
            df_latest = df_latest.groupby('date').agg({'value': 'mean'})

        # convert the processed data to JSON and make a request to the serving predictor function
        jsonstr = df_latest.reset_index().to_json(orient='records')
        arr = json.loads(jsonstr)
        res = serving_fn.invoke(path="/v2/models/parcheggi_predictor_model/infer", body={"inputs": arr})
        res_df = pd.DataFrame(res['outputs'])
        res_df['datetime'] = res_df['date'].astype('datetime64[ms, UTC]')
        res_df['parcheggio'] = parking_str
        res_df['predicted_mean'] = res_df['value']
        res_df = res_df.drop(columns=['date', 'value'])
        pred_df = pd.concat([pred_df, res_df], ignore_index=True)

    # concatenate the predicted results with the existing data (if any) and remove duplicates
    old_pd = pred_df
    try: 
        dat_old = mlrun.get_dataitem('store://datasets/parcheggi/parking_prediction_nbeats_model')
        old_pd = pd.concat([dat_old.as_df(), pred_df], ignore_index=True)
        old_pd = old_pd.drop_duplicates(subset=['parcheggio', 'datetime'])
    except: pass

    # log the predictions as a dataset in the project
    project.log_dataset('parking_prediction_nbeats_model', old_pd, stats=True)

    old_pd = pred_df.copy()
    old_pd['slice_datetime'] = date_str
    try:
        dat_old = mlrun.get_dataitem('store://datasets/parcheggi/parking_prediction_nbeats_model_sliced')
        old_pd = pd.concat([dat_old.as_df(), old_pd], ignore_index=True)
    except: pass

    # log the predictions as a dataset in the project
    project.log_dataset('parking_prediction_nbeats_model_sliced', old_pd, stats=True)
