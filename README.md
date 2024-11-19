# Parking Data

The project aims at elaborating the parking data, creating the necessary datasets, visualizations, and services for historical, actual, and forecasted parking data.
The notebooks on project root demonstrate the functionality of the platform while the generated python source scripts can be found in 'src' folder.

## 1. Data Management

The data management consists of the following activities
- download the historical data and create the dataset on the platform Datalake 
- aggregate the data according by the calendar day and parking for further analysis, saving the dataset in the Datalake and in the database

The data management operations are defined inside the jpyter notebook [1_parcheggi_data_pipeline.ipynb](https://github.com/tn-aixpa/parkings-demo/blob/main/1_parcheggi_data_pipeline.ipynb) which demonstrates a workflow created within the scope digitalhub platform that downloads the storical data, aggregates it, extract the basic parking info. The created data (parkings, aggregated occupation) is stored in the datalake and DB.

## 2. Data Training

The data training consist of following activities
- use of the digital hub platform ambient to read the data from data lake as created in the data management step.
- train the data prediction model using deep learning(NBEATS) and linear regression (SARIMAX) approaches.
- model serve.

In this scenario data is trained and comparison of predictions is drawn between two different approaches

### 2.1 ML Model Training and Serving (NBEATS)

Based on the historical data collected the project allows for training a parking occupancy prediction model and expose it as a service. The training steps demonstrated in the jypter notebook [2_parcheggi_ml_service.ipynb](https://github.com/tn-aixpa/parkings-demo/blob/main/2_parcheggi_ml_service.ipynb). In this step a NBEATS DL global model is created using [darts](https://unit8co.github.io/) framework. The model is then deployed using Nuclio serverless platform with a custom service demonstrated this model to make predictions. The jypter notebook contains the necessary details for declaring a serving functions, adding the trained model and deploying it on the platform using the custom server implementation. The predictions are generated using the deployed serving model at different time slices. The jypter notebook [4_parcheggi_nbeats_model_pipeline.ipynb](https://github.com/tn-aixpa/parkings-demo/blob/main/4_parcheggi_nbeats_model_pipeline.ipynb) demonstrates the step by step process of prediction generation in a scheduled workflow using the digitalhub platform services.

### 2.2 Linear Regerssion Model Training (SARIMAX)
Based on data generated in first step the project allows for training of the SARIMAX (Seasonal Autoregressive Integrated Moving Average + exogenous variables) model which is a  powerful tool for modeling and forecasting both trends and seasonal variations in temporal data. The model in this cased is persisted in the ``digitalhub`` database inside a table named 'parkings_prediction'. The jypter notebook [3_parcheggi_sarimax_regression_pipeline](https://github.com/tn-aixpa/parkings-demo/blob/main/3_parcheggi_sarimax_regression_pipeline.ipynb)) demonstrate the step by step process of forecast generation in a scheduled workflow using the digitalhub platform services.

## 3. Data Visualization

The folder ``dashboards`` contains the dashboards that can be used for the visualization of the parking data. In particular

- ``grafana_parkings.json`` defines a Grafana dashboard with the parking data. Import it in the grafana instance of interest and configure the variable Id in ``Settings`` panel which corresponds to the ``digitalhub`` datasource of your instance.

- ``grafana_parkings_d2.json`` defines a Grafana dashboard with the parking data and graph panel to draw comparision among different predictions and acutal data. Import it in the grafana instance of interest and configure the variable Id in ``Settings`` panel which corresponds to the ``digitalhub`` datasource of your instance.

- ``grafana_parkings_d3_slices`` defines a Grafana dashboard with the parking data and a graph panel that draw specific comparison of predictions generated using deep learning approach at different time slices and acutal data depicting how accurate are the predictions indicated with blue lines with actual data indicated with green line. Import it in the grafana instance of interest and configure the variable Id in ``Settings`` panel which corresponds to the ``digitalhub`` datasource of your instance. 

## 4. Data Services

It is possible to expose APIs for the collected data using [PostgREST](https://postgrest.org/en/stable/) operator. In this way the DB tables may be accessed using the 
corresponding APIs and parameters.

Please note, that in this scenario the data is written to the ``public`` schema of the ``digitalhub`` database. It is recommended to expose explicitly only the specific datasets via dedicated schema and views. For this the following stepls should be performed:

- create a new ``api`` schema for your database (it is possible to use e.g., [SQLPad](https://scc-digitalhub.github.io/docs/components/sqlpad/) workspace from Coder).
```sql
create schema if not exists api;
```
- create views for the stored tables (e.g., using the SQLPad tool as before)
```sql
create or replace view api.parkings as select * from public.parkings;
create or replace view api.parking_data_aggregated as select * from public.parking_data_aggregated;
```
- create a PostgREST instance using the [KRM UI](https://scc-digitalhub.github.io/docs/tasks/resources/#managing-postgrest-data-services-with-krm). Specify ``api`` as exposed schema and an appropriate user (e.g., default read-only user for the ``digitalhub`` database).

- expose the service using API Gateway in [KRM](https://scc-digitalhub.github.io/docs/tasks/resources/#exposing-services-externally)). To access the data represented in the specific view, use the calls ``<host>/<view_name>``. Please take into account that the propagation of the DNS name may require some time. 

