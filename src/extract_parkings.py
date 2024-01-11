
import mlrun
import pandas as pd
import requests

@mlrun.handler(outputs=["parkings"])
def extract_parkings(context, di: mlrun.DataItem):
    KEYS = ['parcheggio', 'lat', 'lon', 'posti_totali']
    df_parcheggi = di.as_df().groupby(['parcheggio']).first().reset_index()[KEYS]
    return df_parcheggi
