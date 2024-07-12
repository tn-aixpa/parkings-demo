from digitalhub_runtime_python import handler
import pandas as pd

@handler(outputs=["parkings"])
def extract_parkings(project, di):
    KEYS = ['parcheggio', 'lat', 'lon', 'posti_totali']
    df_parcheggi = di.as_df().groupby(['parcheggio']).first().reset_index()[KEYS]
    return df_parcheggi
