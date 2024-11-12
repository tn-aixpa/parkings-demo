
import streamlit as st
import digitalhub as dh
import pandas as pd
import dateutil.parser as dparser

PROJECT_NAME = "parcheggi-scheduler"
proj = dh.get_or_create_project(PROJECT_NAME)
print("created project {}".format(PROJECT_NAME))

prediction_nbeats = proj.get_dataitem("parking_prediction_nbeats_model")
prediction_sarimax_regression = proj.get_dataitem("parking_data_predicted_regression")

rdf1 = prediction_sarimax_regression.as_df();
rdf1['datetime'] = rdf1['datetime'].apply(dparser.parse,fuzzy=True)

rdf2 = prediction_nbeats.as_df()
rdf2.datetime = pd.to_datetime(rdf2.datetime, unit='ms')

st.write("""prediction Sarimax regression""")
st.line_chart(rdf1, x="datetime", y="predicted_mean")

st.write("""prediction Nbeats model""")
st.line_chart(rdf2, x="datetime", y="predicted_mean")
