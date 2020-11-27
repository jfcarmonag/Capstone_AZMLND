import xgboost as xgb 
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error, confusion_matrix
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

# TODO: Create TabularDataset using TabularDatasetFactory
# Data is located at:
_DATA_URL = "https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data/download"

def clean_data(data):
    
    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()
    y_df = x_df.pop("Label")
    return x_df, y_df


def get_clean_data():
    ds = TabularDatasetFactory.from_delimited_files(path=_DATA_URL)
    return clean_data(ds)


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    # parser.add_argument('--C', type=float, default=1.0, 
    # help="Inverse of regularization strength. 
    # Smaller values cause stronger regularization")
    # parser.add_argument('--max_iter', type=int, default=100, 
    # help="Maximum number of iterations to converge")
    parser.add_argument('--max_depth', type=float, default= 4.0)
    parser.add_argument('--learning_rate', type=float, default= 0.1)
    parser.add_argument('--gamma', type=float, default= 0.5)
    parser.add_argument('--reg_lambda', type=float, default= 1.0)
    parser.add_argument('--scale_pos_weight', type=float, default=3.0)
    args = parser.parse_args()

    run = Run.get_context()

    run.log("max_depth:", np.float(args.max_depth))
    run.log("learning_rate:", np.int(args.learning_rate))
    run.log("gamma:", np.float(args.gamma))
    run.log("reg_lambda:", np.int(args.reg_lambda))
    run.log("scale_pos_weight:", np.float(args.scale_pos_weight))

    x, y = get_clean_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, 
                                                        random_state=42)
    model = xgb.XGBClassifier(objective="binary:logistic", 
                            missing=Nonemax_depth=args.max_depth, 
                            learning_rate=args.learning_rate, 
                            gamma=args.gamma, reg_lambda=args.reg_lambda,
                            scale_pos_weight=args.scale_pos_weight).fit(x_train,
                            y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    value = {
       "schema_type": "confusion_matrix",
       "schema_version": "v1",
       "data": {
           "class_labels": ["0", "1"],
           "matrix": confusion_matrix(y_test, model.predict(x_test)).tolist()
       }
    }
    run.log_confusion_matrix(name='Confusion Matrix', value=value)
    os.makedirs('outputs', exist_ok=True)
    # note file saved in the outputs folder is automatically uploaded 
    # into experiment record
    joblib.dump(value=model, filename='outputs/model.pkl')

if __name__ == '__main__':
    main()
