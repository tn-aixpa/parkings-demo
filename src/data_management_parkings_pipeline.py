
from digitalhub_runtime_kfp.dsl import pipeline_context

def myhandler(url):
    with pipeline_context() as pc:
        s1_dataset = pc.step(name="download", function="downloader-funct", action="job",inputs={"url":url},outputs={"dataset":"dataset"})        
        s2_parking = pc.step(name="extract_parking", function="extract-parkings", action="job",inputs={"di":s1_dataset.outputs['dataset']},outputs={"parkings":"parkings"})        
        s3_aggregate = pc.step(name="aggregate",  function="aggregate-parkings", action="job",inputs={"di":s1_dataset.outputs['dataset']},outputs={"parking_data_aggregated":"parking_data_aggregated"})        
        s4_predict = pc.step(name="predict-day-sarimax-regression", function="predict-day-sarimax-regression", action="job", inputs={"parkings_di":s1_dataset.outputs['dataset']}, outputs={})
        s5_to_db = pc.step(name="to_db",  function="to-db", action="job",inputs={"agg_di": s3_aggregate.outputs['parking_data_aggregated'],"parkings_di":s2_parking.outputs['parkings']},outputs={})
