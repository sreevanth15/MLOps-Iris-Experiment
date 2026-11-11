import pandas as pd
import joblib
import json
from sklearn.metrics import accuracy_score, f1_score

test_df = pd.read_csv("data/processed/test.csv")

X_test = test_df.drop(columns=["target"])
y_test = test_df["target"]

model = joblib.load("models/iris_model.joblib")
predictions = model.predict(X_test)

metrics = {
    "accuracy": float(accuracy_score(y_test, predictions)),
    "f1_score": float(f1_score(y_test, predictions, average="macro"))
}

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("Evaluation completed.")
print(metrics)
