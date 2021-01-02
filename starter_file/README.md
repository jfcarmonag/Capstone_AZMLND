

# Prediction of Bank Churners using AzureML

*TODO:* In this project we show how to train and deploy a model that predicts bank churners. We use AutoML and Hyperdrive to build two predictive models and choose the best one for deployment. Finally, we save the model in ONNX format for further uses.


## Dataset

### Overview
The dataset contains the features from about 10.000 credit card users. Among the features we find, for example,  The target column is called *Attrition_label*. This data is originally extracted from [Analyttica Website](https://leaps.analyttica.com/home)

### Task
The task in this project is pretty straightforward: we would like to predict the *Attrition_Flag* column. This column contains only two possible values, (after preprocessing, these are: 1=churner, 0=no churner). Thus, this a classification problem of supervised learning type.

### Access
To access the data, we first saved the data in a public [GitHub repository](https://github.com/jfcarmonag/Capstone_AZMLND/raw/master/starter_file/BankChurners.csv), then we use the **TabularDatasetFactory** module to access it.

## Automated ML
The parameters for automl settings and cofiguration are the standard ones as suggested in [Azure AutoML Example](https://docs.microsoft.com/en-us/python/api/azureml-train-automl-client/azureml.train.automl.automlconfig.automlconfig?view=azure-ml-py). The important parameters are:
- "primary_metric": 'accuracy'.
- task = 'classification'
- label_column_name = "Attrition_Flag"

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
