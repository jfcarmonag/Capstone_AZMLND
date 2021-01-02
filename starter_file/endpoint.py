import requests
import json

# URL for the web service, should be similar to:
# 'http://8530a665-66f3-49c8-a953-b82a2d312917.eastus.azurecontainer.io/score'
scoring_uri = ''
# If the service is authenticated, set the key or token
key = ''

# Two sets of data to score, so we get two results back
data = {"data":
        [
          {"Attrition_Flag":0,
          "Customer_Age": 45,
          "Gender_M":1,
          "Gender_F":0,
          "Dependent_count":3,
          "Education_Level":2,
          "Married":0,
          "Single":1,
          "Divorced":0,
          "Unknown":0,
          "Income_Category":2,
          "Card_Category":2,
          "Months_on_book":39,
          "Total_Relationship_Count":5,
          "Months_Inactive_12_mon":1,
          "Contacts_Count_12_mon":3,
          "Credit_Limit":12691,
          "Total_Revolving_Bal":777,
          "Avg_Open_To_Buy":11914,
          "Total_Amt_Chng_Q4_Q1":1.335,
          "Total_Trans_Amt":1144,
          "Total_Trans_Ct":42,
          "Total_Ct_Chng_Q4_Q1":1.625,
          "Avg_Utilization_Ratio":0.061,
          },
      ]
    }
# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())


