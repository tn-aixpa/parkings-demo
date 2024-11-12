from digitalhub_runtime_python import handler
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
import datetime as dtt
import os

@handler()
def to_db(project, agg_di , parkings_di ):
    USERNAME = os.getenv("POSTGRES_USER")
    PASSWORD = os.getenv("POSTGRES_PASSWORD")
    engine = create_engine('postgresql+psycopg2://'+USERNAME+':'+PASSWORD+'@database-postgres-cluster/digitalhub')
    
    agg_df = agg_di.as_df(file_format="parquet")
        
    # Keep only last two calendar years
    date = dtt.date.today() - dtt.timedelta(days=365*2)
    agg_df['day'] = agg_df['day'].apply(lambda t: datetime.fromtimestamp(t)) #added because before was converted the type
    agg_df = agg_df[agg_df['day'].dt.date >= date]
    
    with engine.connect() as connection: 
        try: connection.execute("DELETE FROM parkings, parking_data_aggregated") 
        except: pass

    agg_df.to_sql("parking_data_aggregated", engine, if_exists="append")
    parkings_di.as_df().to_sql('parkings', engine, if_exists="append")
    return
