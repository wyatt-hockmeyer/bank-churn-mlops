# test_api.py
# Sends two sample customers to the running churn API and prints its predictions

import requests

url = "http://127.0.0.1:8000"

# Confirm the server is reachable before sending predictions
response = requests.get(url + "/")
print("Health check :", response.status_code, response.json())

# Matches the highest-risk patterns from the analysis
high_risk = {
    "CreditScore": 600, "Geography": "Germany", "Gender": "Female",
    "Age": 52, "Tenure": 3, "Balance": 125000.0, "NumOfProducts": 3,
    "HasCrCard": 1, "IsActiveMember": 0, "EstimatedSalary": 90000.0,
}

# Matches the lowest-risk patterns from the analysis
low_risk = {
    "CreditScore": 720, "Geography": "France", "Gender": "Male",
    "Age": 32, "Tenure": 6, "Balance": 0.0, "NumOfProducts": 2,
    "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 60000.0,
}

response = requests.post(url + "/predict", json=high_risk)
print("High-risk customer :", response.status_code, response.json())

response = requests.post(url + "/predict", json=low_risk)
print("Low-risk customer :", response.status_code, response.json())