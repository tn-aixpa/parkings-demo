
from statsmodels.tsa.statespace.sarimax import SARIMAX
import mlrun
import pandas as pd
from sqlalchemy import create_engine
import datetime
from pandas.tseries.offsets import DateOffset
import os

def to_point(point):
    """
    Convert a decimal number representing minutes to a datetime object.

    Args:
        point (float): The decimal number representing minutes.

    Returns:
        datetime.datetime: A datetime object representing the current date and time with the minutes derived from the input.

    Example:
        >>> to_point(45)
        datetime.datetime(2022, 1, 1, 0, 22, 30)
    """
    today = datetime.datetime.today()
    return datetime.datetime(today.year, today.month, today.day, int(point * 30 / 60), int(point * 30 % 60))

@mlrun.handler(outputs=["parking_data_predicted"])
def predict_day(context, parkings_di: mlrun.DataItem):
    """
    Predicts the occupancy of parking spaces for the next 48 steps and saves the results in a PostgreSQL database.

    Args:
        context (mlrun.MLClientCtx): The MLRun context object.
        parkings_di (mlrun.DataItem): The data item containing the parking data.

    Returns:
        None
    """
    # Convert data item to pandas DataFrame
    df = parkings_di.as_df()

    # Create a clean copy of the DataFrame
    df_clean = df.copy()

    # Remove unnecessary columns
    df_clean = df_clean.drop(columns=['lat', 'lon'])

    # Convert 'data' column to datetime
    df_clean.data = df_clean.data.astype('datetime64[ms, UTC]')

    # Calculate the occupancy rate
    df_clean['occupied'] = df_clean.posti_occupati / df_clean.posti_totali

    # Round the 'data' column to the nearest 30 minutes
    df_clean['date_time_slice'] = df_clean.data.dt.round('30min').dt.tz_convert(None)

    # Extract the date from the 'data' column
    df_clean['date'] = df_clean.data.dt.date

    # Filter out data from the last 30 days
    df_clean = df_clean[df_clean.date_time_slice >= (datetime.datetime.today() - pd.DateOffset(30))]

    # Filter out data from today
    df_clean = df_clean[df_clean.date <= (datetime.datetime.today() - pd.DateOffset(1))]

    # Remove the 'date' column
    df_clean = df_clean.drop(['date'], axis=1)

    # Ensure that 'posti_occupati' is within the range of [0, posti_totali]
    df_clean.posti_occupati = df_clean.apply(lambda x: max(0, min(x['posti_totali'], x['posti_occupati'])), axis=1)

    # Recalculate the occupancy rate
    df_clean['occupied'] = df_clean.posti_occupati / df_clean.posti_totali

    # Get unique parking locations
    parcheggi = df_clean['parcheggio'].unique()

    # Initialize a list to store the predictions
    res = []

    # Iterate over each parking location
    for parcheggio in parcheggi:
        # Create a copy of the cleaned DataFrame
        cp = df_clean.copy()

        # Filter data for the current parking location
        parc_df = cp[cp['parcheggio'] == parcheggio]

        # Group data by 'date_time_slice' and aggregate metrics
        parc_df = parc_df.groupby('date_time_slice').agg({'posti_occupati':['sum','count'], 'posti_totali':['sum','count']})

        # Calculate the occupancy rate
        parc_df['occupied'] = parc_df.posti_occupati['sum'] / parc_df.posti_totali['sum']

        # Remove unnecessary columns
        parc_df.drop(columns=['posti_occupati', 'posti_totali'], inplace=True)

        # Sort the DataFrame by index
        parc_df.sort_index(inplace=True)

        # Extract the 'occupied' column as a Series
        data = parc_df.reset_index()['occupied']

        # Define the SARIMA model parameters
        my_seasonal_order = (1, 1, 1, 48)

        # Create and fit the SARIMA model
        sarima_model = SARIMAX(data, order=(1, 0, 1), seasonal_order=my_seasonal_order)
        results_SAR = sarima_model.fit(disp=-1)

        # Generate predictions for the next 48 steps
        pred = results_SAR.forecast(steps=48).reset_index()

        # Add the 'parcheggio' column
        pred['parcheggio'] = parcheggio
        res.append(pred)
    
    for pred in res:
        pred['point'] = (pred.index).astype('int')
        pred['datetime'] = pred['point'].apply(to_point)
        pred.drop(['point'], axis=1, inplace=True)
    
    all = pd.concat(res, ignore_index=True)[['predicted_mean', 'parcheggio', 'datetime']]
    
    USERNAME = context.get_secret('DB_USERNAME')
    PASSWORD = context.get_secret('DB_PASSWORD')
    engine = create_engine('postgresql://'+USERNAME+':'+PASSWORD+'@database-postgres-cluster/digitalhub')
    with engine.connect() as connection: 
        try: connection.execute("DELETE FROM parkings_prediction")
        except: pass

    all.to_sql('parkings_prediction', engine, if_exists="append")

    old_pd = all
    try: 
        dat_old = mlrun.get_dataitem('parking_prediction_sarima_model')
        old_pd = pd.concat([dat_old.as_df(), all], ignore_index=True)
    except: pass
    context.log_dataset('parking_prediction_sarima_model', old_pd, stats=True)
    return all