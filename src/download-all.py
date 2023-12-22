
import mlrun
import pandas as pd
import requests

@mlrun.handler(outputs=["dataset"])
def downloader(context, url: mlrun.DataItem):
    # read and rewrite to normalize and export as data
    df = url.as_df(format='csv',sep=";")
    df[['lat', 'lon']] = df['coordinate'].str.split(', ',expand=True)
    df = df.drop(columns=['% occupazione', 'GUID', 'coordinate']).rename(columns={'Parcheggio': 'parcheggio', 'Data': 'data', 'Posti liberi': 'posti_liberi', 'Posti occupati': 'posti_occupati', 'Posti totali': 'posti_totali'})
    df["lat"] = pd.to_numeric(df["lat"])
    df["lon"] = pd.to_numeric(df["lon"])

    return df
