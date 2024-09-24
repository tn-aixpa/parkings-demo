from digitalhub_runtime_python import handler
import pandas as pd

@handler(outputs=["dataset"])
def downloader(project, url):
    df = url.as_df(file_format='csv',sep=";")
    df[['lat', 'lon']] = df['coordinate'].str.split(', ',expand=True)
    df = df.drop(columns=['% occupazione', 'GUID', 'coordinate']).rename(columns={'Parcheggio': 'parcheggio', 'Data': 'data', 'Posti liberi': 'posti_liberi', 'Posti occupati': 'posti_occupati', 'Posti totali': 'posti_totali'})
    df["lat"] = pd.to_numeric(df["lat"])
    df["lon"] = pd.to_numeric(df["lon"])
    return df
