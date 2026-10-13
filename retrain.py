# retrain.py
# Checks the deployed model's accuracy on the newest customer batch and retrains it
# when accuracy falls below a threshold (documented exception: accuracy-threshold retraining)

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

accuracy_threshold = 0.75

# Rebuild the original training data exactly as in the notebook (Sections 3 and 8)
df = pd.read_csv("data/Churn_Modeling.csv")
df = df.rename(columns={"CustomerId": "RowNumber", "CustomerId.1": "CustomerId"})
X = df.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"])
y = df["Exited"]
X = pd.get_dummies(X, columns=["Geography", "Gender"], drop_first=True, dtype="int")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Newest batch from monitoring; half joins the training data, half is held back
# so the current and retrained models are judged on customers neither trained on
new_batch = pd.read_csv("data/new_customers.csv")
new_X = new_batch.drop(columns=["Exited"])
new_y = new_batch["Exited"]
new_train_X, new_eval_X, new_train_y, new_eval_y = train_test_split(
    new_X, new_y, test_size=0.5, random_state=42
)

# Step 1: check the deployed model against the threshold
current_model = joblib.load("models/churn_model.pkl")
current_accuracy = accuracy_score(new_eval_y, current_model.predict(new_eval_X))
print("Current model accuracy on new data :", round(current_accuracy, 3))
print("Accuracy threshold :", accuracy_threshold)

if current_accuracy >= accuracy_threshold:
    print("Accuracy meets the threshold; no retraining needed")
else:
    print("Accuracy is below the threshold; retraining")

    # Step 2: retrain the same model type on the original plus new training data
    combined_X = pd.concat([X_train, new_train_X])
    combined_y = pd.concat([y_train, new_train_y])
    new_model = ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("model", GradientBoostingClassifier(random_state=42))
    ])
    new_model.fit(combined_X, combined_y)

    # Step 3: compare on the held-back new customers, and confirm the original test set still scores well
    new_accuracy = accuracy_score(new_eval_y, new_model.predict(new_eval_X))
    new_f1 = f1_score(new_eval_y, new_model.predict(new_eval_X))
    original_accuracy = accuracy_score(y_test, new_model.predict(X_test))
    print("Retrained model accuracy on new data :", round(new_accuracy, 3))
    print("Retrained model F1 on new data :", round(new_f1, 3))
    print("Retrained model accuracy on original test set :", round(original_accuracy, 3))

    # Step 4: replace the deployed model only if the retrained one is better
    if new_accuracy > current_accuracy:
        joblib.dump(new_model, "models/churn_model.pkl")

        # Log the retraining run and register it as a new version (Lesson 38 Demo 3 pattern)
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("bank-churn")
        with mlflow.start_run(run_name="Retrained Gradient Boosting + SMOTE"):
            mlflow.log_param("accuracy_threshold", accuracy_threshold)
            mlflow.log_metric("previous_accuracy_new_data", current_accuracy)
            mlflow.log_metric("accuracy_new_data", new_accuracy)
            mlflow.log_metric("f1_new_data", new_f1)
            mlflow.log_metric("accuracy_original_test", original_accuracy)
            mlflow.sklearn.log_model(
                new_model,
                name="model",
                serialization_format="cloudpickle",
                registered_model_name="churn-model",
            )
        print("Retrained model saved to models/churn_model.pkl and registered as a new churn-model version")
    else:
        print("Retrained model did not improve; keeping the current model")