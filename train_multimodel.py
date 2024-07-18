
import mlrun
import pandas as pd

from darts import TimeSeries

from darts.models import NBEATSModel
from darts.metrics import mape, smape, mae
from darts.dataprocessing.transformers import Scaler
from zipfile import ZipFile

import logging
logging.disable(logging.CRITICAL)

import warnings
warnings.filterwarnings("ignore")

import datetime

import pandas as pd
import datetime

from pickle import dumps

def fill_missing_days(parc_df):
    """
    Add one occurrence in the df of complete days that are missing from the data,
    this occurrence it's then used in fill_missing(),
    in the missing_days array it is possible to see all the days that were missing from the data
    """
    temp = pd.Series(parc_df.index.date).value_counts()  # Count the occurrences of each date
    temp.sort_index(inplace=True)
    lowest_date = temp.index[0]
    highest_date =  temp.index[-1]
    days_total = (highest_date - lowest_date).days 
    actual_days = len(temp.index)
    start = lowest_date
    missing_days = []
    if days_total !=actual_days:
        for i in range(actual_days):
            t = start+datetime.timedelta(days=i)
            if t not in temp.index:
                missing_days.append(t)
                ts = datetime.datetime(t.year, t.month, t.day, 0, 0)
                for i in range(24):
                    for j in range(0,30):
                        a = datetime.datetime(temp.index[0].year, temp.index[0].month, temp.index[0].day, i, j)
                        if a in parc_df.index:
                            parc_df.loc[ts] = parc_df.loc[a].copy()
    return missing_days
    

def fill_missing(parc_df):
    """
    Givend the DF if there are dates with less occurrences of 48 (2 occurrence for each hour of the day)
    then if first check if there is data for the same day 1 week prior or after if it does not exists it check 2 weeks prior or after
    if still does not exists it happends the missing data to an array
    """
    missing = []  # List to store timestamps for which values could not be filled
    temp = pd.Series(parc_df.index.date).value_counts()  # Count the occurrences of each date
    temp.sort_index(inplace=True)
    temp = temp[temp < 48]  # Filter dates with less than 48 occurrences
    temp.sort_index(inplace=True)  # Sort the dates in ascending order
    for t in temp.index:  # Iterate through the filtered dates
        for h in range(24):  # Iterate through 24 hours
            for half_hour in [0, 30]:  # Iterate through 0 and 30 minutes
                ts = datetime.datetime(t.year, t.month, t.day, h, half_hour)  # Create a timestamp
                if ts not in parc_df.index:  # If the timestamp is missing in the DataFrame
                    if ts - datetime.timedelta(days=7) in parc_df.index:  # Check if the previous week's timestamp is available
                        parc_df.loc[ts] = parc_df.loc[ts - datetime.timedelta(days=7)].copy()  # Copy values from the previous week
                    elif ts + datetime.timedelta(days=7) in parc_df.index:  # Check if the next week's timestamp is available
                        parc_df.loc[ts] = parc_df.loc[ts + datetime.timedelta(days=7)].copy()  # Copy values from the next week
                    elif ts - datetime.timedelta(days=14) in parc_df.index:  # Check if the previous 2 week's timestamp is available
                        parc_df.loc[ts] = parc_df.loc[ts - datetime.timedelta(days=14)].copy()  # Copy values from the previous week
                    elif ts + datetime.timedelta(days=14) in parc_df.index:  # Check if the next 2 week's timestamp is available
                        parc_df.loc[ts] = parc_df.loc[ts + datetime.timedelta(days=14)].copy()  # Copy values from the next week
                    else:
                        missing.append(ts)  # If values cannot be filled, add the timestamp to the missing list
    return missing

def get_inference(missing_data,parc_df):
    """
    Given a list of missing data and a DF, this procedure will first find a window of missing data
    it will try to complete that window inferring the values in between taking the first and last value closest
    to the window and then calculating the values of the points in betweens
    """
    date_time_slice = []
    occupied = []
    hour = []
    dow=[]
    while len(missing_data)>0:
        start = missing_data.pop(0) #takes the first 
        end= start
        steps = 1
        while len(missing_data)>0 and end + datetime.timedelta(minutes=30) == missing_data[0]:
            #find windows missing data
            end =missing_data.pop(0)
            steps+=1
        value_start = None
        value_end = None
        if start - datetime.timedelta(minutes=30) in parc_df.index:
            value_start =  parc_df.loc[start - datetime.timedelta(minutes=30)]['occupied']
        if end + datetime.timedelta(minutes=30) in parc_df.index:
            value_end =  parc_df.loc[end + datetime.timedelta(minutes=30)]['occupied']
        incremental_step = 0
        if value_start==None and value_end == None:
            # MISSING both data before and after the window found hopefully doesn't happens
            raise Exception("This dataset has too many holes we suggest to delete this dataset for the training")
        elif value_start == None:
            value_start = value_end
            incremental_step = 0
        elif value_end == None:
            value_end=value_start
            incremental_step = 0
        else:
            incremental_step = (value_end-value_start)/(steps+1)
        for i in range(steps):
            ts = start + i*datetime.timedelta(minutes=30)
            vval = value_start + (i+1)*incremental_step
            date_time_slice.append(ts)
            occupied.append(vval)
    result = {"date_time_slice":date_time_slice,"occupied":occupied}
    return pd.DataFrame(result,index=date_time_slice)


def fill_all_missing(parc_df,inference=False):
    """
    Given a DF it fills the days and occurrences missings in the data, it is possible to enable inference if 
    the DF has still holes after the filling of the missing data, but with inference enabled the data quality it's lower 
    due to artificial created data under euristics
    """
    # enable inference if the data still miss some parts
    missing_days = fill_missing_days(parc_df)
    #print(f"Missing days found with zero data: {missing_days}")
    missing_parts = fill_missing(parc_df)
    n_parc_df = parc_df
    if len(missing_parts)>0 and inference:
        inference_df = get_inference(missing_parts,parc_df)
        inference_df['hour'] = inference_df.date_time_slice.dt.hour
        inference_df['dow'] = inference_df.date_time_slice.dt.dayofweek
        inference_df = inference_df.drop(columns=['date_time_slice'])
        n_parc_df = pd.concat([parc_df,inference_df])
        n_parc_df.sort_index(inplace=True)
    return n_parc_df

@mlrun.handler()
def train_model(context, parkings_di: mlrun.DataItem, window: int = 60, 
                input_chunk_length: int = 24, output_chunk_length: int = 12, n_epochs: int = 100, 
                split_ratio: float = 0.8):

    # Load the input data
    df_source = parkings_di.as_df()

    # Clean the data
    df_clean = df_source.copy()
    df_clean.data = df_clean.data.astype('datetime64')
    df_clean['occupied'] = df_clean.posti_occupati / df_clean.posti_totali
    df_clean['date_time_slice'] = df_clean.data.dt.round('30min')
    df_clean = df_clean[df_clean.date_time_slice >= (datetime.datetime.today() - pd.DateOffset(window))]
    df_clean = df_clean[df_clean.date_time_slice <= (datetime.datetime.today() - pd.DateOffset(1))]
    df_clean.posti_occupati = df_clean.apply(lambda x: max(0, min(x['posti_totali'], x['posti_occupati'])), axis=1)
    df_clean['occupied'] = df_clean.posti_occupati / df_clean.posti_totali
    df_clean = df_clean.drop(columns=['lat', 'lon', 'data', 'posti_totali', 'posti_liberi', 'posti_occupati'])
    parcheggi = df_clean['parcheggio'].unique()

    train_sets, val_sets = [], []

    # Process data for each parking lot
    for parcheggio in parcheggi:
        parc_df = df_clean[df_clean['parcheggio'] == parcheggio]
        parc_df['hour'] = parc_df.date_time_slice.dt.hour
        parc_df['dow'] = parc_df.date_time_slice.dt.dayofweek
        parc_df = parc_df.drop(columns=['parcheggio'])
        parc_df = parc_df.groupby('date_time_slice').agg({'occupied': 'mean', 'hour': 'first', 'dow': 'first'})
        parc_df = fill_all_missing(parc_df)
        ts = TimeSeries.from_dataframe(parc_df,  value_cols='occupied', freq='30min')
        ts_scaled = Scaler().fit_transform(ts)
        
        split = int(len(ts_scaled) * (1 - split_ratio))

        # Split data into training and validation sets
        train, val = ts_scaled[:-split], ts_scaled[-split:]
        train_sets.append(train)
        val_sets.append(val)

    # Train a multi-model using the NBEATS algorithm
    multimodel =  NBEATSModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        n_epochs=n_epochs,
        random_state=0
    )

    # Fit the model to the training sets
    multimodel.fit(train_sets)
    pred = multimodel.predict(n=output_chunk_length*2, series=train_sets[0][:-output_chunk_length*2])

    multimodel.save("parcheggi_predictor_model.pt")
    with ZipFile("parcheggi_predictor_model.pt.zip", "w") as z:
        z.write("parcheggi_predictor_model.pt")
        z.write("parcheggi_predictor_model.pt.ckpt")

    # log model to MLRun
    context.log_model(
        "parcheggi_predictor_model",
        parameters={
            "window": window,
            "input_chunk_length": input_chunk_length,
            "output_chunk_length": output_chunk_length,
            "n_epochs": n_epochs,
            "split_ratio": split_ratio,
            "num_layers": multimodel.num_layers,
            "layer_widths": multimodel.layer_widths,
            "activation": multimodel.activation
        },
        metrics = {
            "mape": mape(train_sets[0], pred),
            "smape": smape(train_sets[0], pred),
            "mae": mae(train_sets[0], pred)
        },
        model_file="parcheggi_predictor_model.pt.zip",
        labels={"class": "darts.models.NBEATSModel"},
        algorithm="darts.models.NBEATSModel",
        framework="darts"
    ) 
