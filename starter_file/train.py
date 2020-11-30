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
_DATA_URL = "https://github.com/jfcarmonag/nd00333-capstone/blob/master/starter_file/BankChurners.csv"

def clean_data(data):
    
    # Clean and one hot encode data
    # Cleaning and one-hot encoding extracted from 
    # https://www.kaggle.com/sakshigoyal7/churned-customers-recall-of-86
    x_df = data.to_pandas_dataframe().dropna()
    x_df['Attrition_Flag'].replace({'Existing Customer':0, 
    'Attrited Customer':1},inplace=True)
    df.drop(df.columns[[0,-1,-2]].values,axis=1,inplace=True)
    map_education_level = {'High School':1,'Graduate':3,'Uneducated':0,
    'College':2,'Post-Graduate':4,'Doctorate':5}
    map_income_level = {'$60K - $80K':3,'Less than $40K':1, '$80K - $120K':4,
    '$40K - $60K':2,'$120K +':5}
    map_card_category = {'Blue':1,'Gold':3,'Silver':2,'Platinum':4}
    df['Education_Level'].replace(map_education_level,inplace=True)
    df['Income_Category'].replace(map_income_level,inplace=True)
    df['Card_Category'].replace(map_card_category,inplace=True)
    # #hot encoding of gender category
    df.insert(2,'Gender_M',df['Gender'],True)
    df.rename({'Gender':'Gender_F'},axis=1,inplace=True)
    df['Gender_M'].replace({'M':1,'F':0},inplace=True)
    df['Gender_F'].replace({'M':0,'F':1},inplace=True)
    #
    # #hot encoding of marital status
    df.insert(7,'Single',df['Marital_Status'],True)
    df.insert(7,'Divorced',df['Marital_Status'],True)
    df.insert(7,'Unknown',df['Marital_Status'],True)
    df.rename({'Marital_Status':'Married'},axis=1,inplace=True)
    df['Married'].replace({'Single':0, 'Married':1, 'Divorced':0, 'Unknown':0},
    inplace=True)
    df['Single'].replace({'Single':1, 'Married':0, 'Divorced':0, 'Unknown':0},
    inplace=True)
    df['Divorced'].replace({'Single':0, 'Married':0, 'Divorced':1, 'Unknown':0},
    inplace=True)
    df['Unknown'].replace({'Single':0, 'Married':0, 'Divorced':0, 'Unknown':1},
    inplace=True)
    y_df = x_df.pop("Attrition_Flag")
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
