from cloudpickle import load
import numpy as np
import mlrun
from darts import TimeSeries
import pandas as pd
from darts.models import NBEATSModel
from zipfile import ZipFile

class ParkingPredictorModel(mlrun.serving.V2ModelServer):
    def load(self):
        """
        Load and initialize the model and/or other elements.
        """
        # Get the model file and any extra data
        model_file, extra_data = self.get_model('.zip')
        # Open the model file as a zip file
        file = ZipFile(model_file)
        # Get the list of members (files) in the zip file and extract all of them to /tmp/model
        members = file.namelist()
        file.extractall('/tmp/model')
        # Load the NBEATS model from the first member file
        self.model = NBEATSModel.load('/tmp/model/'+members[0])
    
    def predict(self, body: dict) -> list:
        """Generate model predictions from input data.

        Args:
        - body: A dictionary containing input data.

        Returns:
        - A list of model predictions.
        """
        # Create a TimeSeries from the input data
        ts = TimeSeries.from_dataframe(
            pd.DataFrame(body['inputs']),
            time_col="date",
            value_cols="value",
            freq="30min"
        )

        # Generate predictions using the model
        result = self.model.predict(12, series=ts)

        # Convert the result to a pandas DataFrame, reset the index, and convert to a list of dictionaries
        return result.pd_dataframe().reset_index().to_dict('records')
