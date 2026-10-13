# evidently_monitoring.py
# Simulates production monitoring with Evidently AI: compares new customer data
# against the training baseline and reports data drift and model accuracy

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

# Rebuild the exact features and split used in the notebook (Sections 3 and 8)
df = pd.read_csv("data/Churn_Modeling.csv")
df = df.rename(columns={"CustomerId": "RowNumber", "CustomerId.1": "CustomerId"})
X = df.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"])
y = df["Exited"]
X = pd.get_dummies(X, columns=["Geography", "Gender"], drop_first=True, dtype="int")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = joblib.load("models/churn_model.pkl")


# Evidently expects the true label in "target" and the model's output in "prediction"
def add_labels(features, labels):
    data = features.copy()
    data["target"] = labels.values
    data["prediction"] = model.predict(features)
    return data


# Reference = training data; current = held-out customers the model has never seen
reference = add_labels(X_train, y_train)
current = add_labels(X_test, y_test)

# Simulated future batch: customers are 10 years older and none are active members,
# a shift the model never saw in training
drifted_features = X_test.copy()
drifted_features["Age"] = drifted_features["Age"] + 10
drifted_features["IsActiveMember"] = 0
drifted = add_labels(drifted_features, y_test)

# Save the simulated batch so the retraining script can check the model against it
new_batch = drifted_features.copy()
new_batch["Exited"] = y_test.values
new_batch.to_csv("data/new_customers.csv", index=False)


# Build one report (Lesson 39 Demo 3 presets), save it, and return its headline numbers
def build_report(current_data, file_name):
    report = Report(metrics=[
        DataDriftPreset(),
        ClassificationPreset()
    ])
    report.run(reference_data=reference, current_data=current_data)
    report.save_html(file_name)

    drift_share = 0
    accuracy = 0
    for metric in report.as_dict()["metrics"]:
        if metric["metric"] == "DatasetDriftMetric":
            drift_share = metric["result"]["share_of_drifted_columns"]
        if metric["metric"] == "ClassificationQualityMetric":
            accuracy = metric["result"]["current"]["accuracy"]
    return drift_share, accuracy


os.makedirs("reports", exist_ok=True)

drift_share, accuracy = build_report(current, "reports/monitoring_report.html")
print("Normal batch")
print("  Share of drifted columns :", round(drift_share, 2))
print("  Accuracy :", round(accuracy, 3))

drift_share, accuracy = build_report(drifted, "reports/drift_simulation_report.html")
print("Simulated drift batch")
print("  Share of drifted columns :", round(drift_share, 2))
print("  Accuracy :", round(accuracy, 3))

print("Reports saved to the reports folder; new batch saved to data/new_customers.csv")