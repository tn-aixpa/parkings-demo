from digitalhub_runtime_python import handler
import datetime 
import requests
import json
import pandas as pd
import digitalhub as dh

def filter30days(df):
    month = datetime.datetime.now() - pd.to_timedelta('30day')
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
    filtered_df = df[df['datetime'] > month]
    filtered_df['datetime'] = filtered_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    return filtered_df
    
@handler()
def predict_day(project,  parkings_di):
    """
    Monitor and predict parking occupancy.
    """

    # get serving predictor function run
    run_serve_model =  project.get_run(identifier='2fab77c9-685d-49b5-a5ef-728358e80cae')
    
    # get current date and time as string
    date_str = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

    # get parkings dataset and convert it to a dataframe
    parkings_df = parkings_di.as_df()

    # initialize an empty dataframe for predictions
    pred_df = pd.DataFrame(columns=['parcheggio', 'datetime', 'predicted_mean'])

    # iterate over each parking in the dataset
    parcheggi =  parkings_df['parcheggio'].unique()
    #parcheggi = ['Riva Reno' ,'VIII Agosto']
    for parking_str in parcheggi:
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
            df_latest.data = df_latest.data.astype('datetime64[ns, UTC]')
            df_latest['value'] = df_latest.posti_occupati / df_latest.posti_totali
            df_latest['date'] = df_latest.data.dt.round('30min')
            df_latest = df_latest.drop(columns=['parcheggio'])
            df_latest = df_latest.groupby('date').agg({'value': 'mean'})

        # convert the processed data to JSON and make a request to the serving predictor function
        jsonstr = df_latest.reset_index().to_json(orient='records')
        arr = json.loads(jsonstr)
        SERVICE_URL = run_serve_model.status.to_dict()["service"]["url"]
        with requests.post(f'http://{SERVICE_URL}', json={"inference_input":arr}) as r:
            res = json.loads(r.content)
        res_df = pd.DataFrame(res)
        res_df['datetime'] = res_df['date']
        res_df['parcheggio'] = parking_str
        res_df['predicted_mean'] = res_df['value']
        res_df = res_df.drop(columns=['date', 'value'])
        pred_df = pd.concat([pred_df, res_df], ignore_index=True)

    # concatenate the predicted results with the existing data (if any) and remove duplicates
    old_pd = pred_df
    try: 
        dat_old = project.get_dataitem('parking_prediction_nbeats_model')
        old_pd = pd.concat([dat_old.as_df(), pred_df], ignore_index=True)
        old_pd = old_pd.drop_duplicates(subset=['parcheggio', 'datetime'])
    except: pass

    # log the predictions as a dataset in the project
    filter_pd = filter30days(old_pd)
    project.log_dataitem('parking_prediction_nbeats_model', data=filter_pd, kind="table")

    old_pd = pred_df.copy()
    old_pd['slice_datetime'] = date_str
    try:
        dat_old = project.get_dataitem('parking_prediction_nbeats_model_sliced')
        old_pd = pd.concat([dat_old.as_df(), old_pd], ignore_index=True)
    except: pass

    # log the predictions as a dataset in the project
    filter_sliced = filter30days(old_pd)
    project.log_dataitem('parking_prediction_nbeats_model_sliced', data=filter_sliced, kind="table")
