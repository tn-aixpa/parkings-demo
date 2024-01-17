# GDB: Parking Data

The project aims at elaborating the parking data, creating the necessary datasets, visualizations, and services for historical, actual, and forecasted parking data.

## 1. Data Management

The data management consists of the following activities
- download the historical data and create the dataset on the platform Datalake
- aggregate the data according by the calendar day and parking for further analysis, saving the dataset in the Datalake and in the database
- continuously download and update the latest data to have it available in the DB for exposure as API and visualizations
- process data in order to create forecast models for the parkings and continously predict the occupation rate for the next 24 hours

The data management operations are captured by the two procedures:
- "extract-parkings-latest": download the latest data and store it in datalake and PostgreSQL DB. The operation is scheduled eacvh 10 minutes.
- "data-update-pipeline": a pipeline that downloads the storical data, aggregates it, extract the basic parking info, creates the predicition models and applies them to forecast next day data. The created data (parkings, aggregated occupation, forecast) is stored in the datalake and DB. The pipeline is then scheduled for execution every day.

To run / schedule the procedures it is possible to follow the "parcheggi" Jupyter notebook. Please note that
to work with the associated source code it is necessary to provide a set of secrets, specifically
- the ``GIT_TOKEN`` to access the private repo to clone the code (necessary also for executions)
- the ``DB_USERNAME`` and ``DB_PASSWORD`` to write the data to the database. 

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
create or replace view api.parkings_latest as select * from public.parkings_latest;
create or replace view api.parkings_prediction as select * from public.parkings_prediction;
```
- create a PostgreSQL instance using the [KRM UI](https://scc-digitalhub.github.io/docs/tasks/resources/#managing-postgrest-data-services-with-krm). Specify ``api`` as exposed schema and an appropriate user (e.g., default read-only user for the ``digitalhub`` database).

- expose the service using API Gateway in [KRM](https://scc-digitalhub.github.io/docs/tasks/resources/#exposing-services-externally)). To access the data represented in the specific view, use the calls ``<host>/<view_name>``. Please take into account that the propagation of the DNS name may require some time. 

## 3. Data Visualization

The folder ``dashboards`` contains the dashboards that can be used for the visualization of the parking data. In particular

- ``grafana_parkings.json`` defines a Grafana dashboard with the parking data. Import it in the grafana instance of interest and replace the UID of the ``digitalhub`` datasource with the one configured in your instance. 