# Parking Data

The project aims at elaborating the parking data, creating the necessary datasets, visualizations, and services for historical, actual, and forecasted parking data.

## 1. Data Management

The data management consists of the following activities
- download the historical data and create the dataset on the platform Datalake
- aggregate the data according by the calendar day and parking for further analysis, saving the dataset in the Datalake and in the database

The data management operations are defined with the "parkings-pipeline" procedure: a pipeline that downloads the storical data, aggregates it, extract the basic parking info. The created data (parkings, aggregated occupation) is stored in the datalake and DB. 

The notebooks demonstrate the functionality of the platform.

## 2. Data Services

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

## 3. Data Visualization

The folder ``dashboards`` contains the dashboards that can be used for the visualization of the parking data. In particular

- ``grafana_parkings.json`` defines a Grafana dashboard with the parking data. Import it in the grafana instance of interest and replace the UID of the ``digitalhub`` datasource with the one configured in your instance. 

## 4. ML Model Training and Serving

Based on the historical data collected the project allows for training a parking occupancy prediction model and expose it as a service. The training code 
is managed by the ``train-multimodel`` operation (see ``src/train_multimodel.py`` for details) and creates a NBEATS DL global model using [darts](https://unit8co.github.io/) framework. The model is then deployed using Nuclio serverless platform with a custom service for this model to make predictions. The "parcheggi" contains the necessary details for declaring a serving functions, adding the trained model and deploying it on the platform using the custom server implementation (see ``predictor_serving.py`` for further details).
