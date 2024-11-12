
from digitalhub_runtime_kfp.dsl import pipeline_context

def myhandler(di):
    with pipeline_context() as pc:
        s2_predict = pc.step(name="predict-day-nbeats-model", function="predict-day-nbeats-model", action="job", inputs={"parkings_di":di}, outputs={})
