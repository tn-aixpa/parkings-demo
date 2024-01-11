
from statsmodels.tsa.statespace.sarimax import SARIMAX
import mlrun
import pandas as pd
from sqlalchemy import create_engine
import datetime
from pandas.tseries.offsets import DateOffset
import os

def to_point(point):
    today = datetime.datetime.today()
    return datetime.datetime(today.year, today.month, today.day, int(7 + point * 20 / 60), int(point * 20 % 60))

@mlrun.handler()
def predict_day(context, parkings_di: mlrun.DataItem):
    df = parkings_di.as_df()
    df_clean = df.copy()
    df_clean = df_clean.drop(columns=['lat', 'lon'])
    df_clean.data = df_clean.data.astype('datetime64')
    df_clean['occupied'] = df_clean.posti_occupati / df_clean.posti_totali
    df_clean['date_time_slice'] = df_clean.data.dt.round('15min')
    df_clean['date'] = df_clean.data.dt.date
    df_clean = df_clean[df_clean.date_time_slice.dt.hour < 23]
    df_clean = df_clean[df_clean.date_time_slice.dt.hour >= 7]
    df_clean = df_clean[df_clean.date_time_slice >= (datetime.datetime.today() - pd.DateOffset(30))]
    df_clean = df_clean[df_clean.date <= (datetime.datetime.today() - pd.DateOffset(1))]
    df_clean = df_clean.drop(['date'], axis=1)
    df_clean.posti_occupati = df_clean.apply(lambda x: max(0, min(x['posti_totali'], x['posti_occupati'])), axis=1)
    df_clean['occupied'] = df_clean.posti_occupati / df_clean.posti_totali

    parcheggi = df_clean['parcheggio'].unique()
    
    res = []
    for parcheggio in parcheggi:
        cp = df_clean.copy()
        parc_df = cp[cp['parcheggio'] == parcheggio]
        parc_df = parc_df.groupby('date_time_slice').agg({'posti_occupati':['sum','count'], 'posti_totali':['sum','count']})
        parc_df['occupied'] = parc_df.posti_occupati['sum'] / parc_df.posti_totali['sum']
        parc_df.drop(columns=['posti_occupati', 'posti_totali'], inplace=True)
        parc_df.sort_index(inplace=True)   
        data = parc_df.reset_index()['occupied']
        my_seasonal_order = (1, 1, 1, 48)
        sarima_model = SARIMAX(data, order=(1, 0, 1), seasonal_order=my_seasonal_order)
        results_SAR = sarima_model.fit(disp=-1)
        pred = results_SAR.forecast(steps=48).reset_index()
        pred['parcheggio'] = parcheggio
        res.append(pred)
    today = datetime.datetime.today()
    for pred in res:
        pred['point'] = (pred.index).astype('int')
        pred['datetime'] = pred['point'].apply(to_point)
        pred.drop(['point'], axis=1, inplace=True)
    
    all = pd.concat(res, ignore_index=True)[['predicted_mean', 'parcheggio', 'datetime']]
    
    USERNAME = context.get_secret('DB_USERNAME')
    PASSWORD = context.get_secret('DB_PASSWORD')
    engine = create_engine('postgresql://'+USERNAME+':'+PASSWORD+'@database-postgres-cluster/digitalhub')
    all.to_sql('parkings_prediction', engine, if_exists="replace")
    return
