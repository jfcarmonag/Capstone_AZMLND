import json
import numpy as np
import os
import pickle
from sklearn.externals import joblib
from sklearn.linear_model import Ridge
from azureml.core.model import Model
import time


def init():
    global model
    #Print statement for appinsights custom traces:
    print ("model initialized" + time.strftime("%H:%M:%S"))
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    model = joblib.load(model_path)

def run(data):
    try:
        data = np.array(json.loads(data))
        result = model.predict(data)
        # You can return any data type, as long as it is JSON serializable.
         # Log the input and output data to appinsights:
        info = {
            "input": raw_data,
            "output": result.tolist()
            }
        print(json.dumps(info))
        return result.tolist()
        
    except Exception as e:
        error = str(e)
        print (error + time.strftime("%H:%M:%S"))
        return error