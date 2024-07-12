from datetime import datetime
from digitalhub_runtime_python import handler
import pandas as pd

@handler(outputs=["parking_data_aggregated"])
def aggregate_parkings(project, di):
    rdf = di.as_df()
    rdf['data'] = pd.to_datetime(rdf['data'])
    rdf['day'] = rdf['data'].apply(lambda t: t.replace(second=0, minute=0))
    rdf['hour'] = rdf['day'].dt.hour
    rdf['dow'] = rdf['day'].dt.dayofweek
    #rdf['type'] = rdf['data']#.apply(lambda t: "sadassad"+t.astype(str))
    rdf['day'] = rdf['day'].apply(lambda t: datetime.timestamp(t)) #added because complain of timestamp not JSOn serializable#
    rdf = rdf.drop(columns=['data'])
    rdf['lat'] = rdf['lat'].apply(lambda t: float(t))
    rdf['lon'] = rdf['lon'].apply(lambda t: float(t))
    grouped = rdf.groupby(['parcheggio','day']).mean() #
    df_aggregated = grouped.reset_index()
    return df_aggregated
