import mlrun
import pandas as pd

from darts import TimeSeries

from darts.models import NBEATSModel
from darts.metrics import mape, smape, mae
from darts.dataprocessing.transformers import Scaler

import logging
logging.disable(logging.CRITICAL)

import warnings
warnings.filterwarnings("ignore")

import datetime

import pandas as pd
import datetime

from pickle import dumps

def fill_missing(parc_df):
    """
    Fill missing values in the given DataFrame by copying values from the nearest
    available dates within a range of 7 days.
    
    Args:
    parc_df (pd.DataFrame): The DataFrame to fill missing values in.
    
    Returns:
    list: List of timestamps for which values could not be filled.
    """
    missing = []  # List to store timestamps for which values could not be filled
    temp = pd.Series(parc_df.index.date).value_counts()  # Count the occurrences of each date
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
                    else:
                        missing.append(ts)  # If values cannot be filled, add the timestamp to the missing list
    return missing  # Return the list of timestamps for which values could not be filled

@mlrun.handler()
def train_model(context, parkings_di: mlrun.DataItem, window: int = 60, 
                input_chunk_length: int = 24, output_chunk_length: int = 12, n_epochs: int = 100, 
                split_ratio: float = 0.8):
    """
    Trains a model using the N-BEATS algorithm.

    Parameters:
    context (mlrun.MLClientCtx): The context object for the function.
    parkings_di (mlrun.DataItem): The input data for training.
    window (int): The size of the time window for data aggregation.
    horizon (int): The forecast horizon.
    input_chunk_length (int): The input chunk size for the model.
    output_chunk_length (int): The output chunk size for the model.
    n_epochs (int): The number of epochs for training.
    split_ratio (float): The ratio for splitting the dataset into training and validation sets.
    """

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
        fill_missing(parc_df)
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

    # log model to MLRun
    context.log_model(
        "parcheggi_predictor_model",
        body=dumps(multimodel),
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
        model_file="parcheggi_predictor_model.pkl",
        labels={"class": "darts.models.NBEATSModel"},
        algorithm="darts.models.NBEATSModel",
        framework="darts"
    ) 
