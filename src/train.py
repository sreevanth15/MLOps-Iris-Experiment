import pandas as pd
import yaml
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

train_df = pd.read_csv("data/processed/train.csv")

X_train = train_df.drop(columns=["target"])
y_train = train_df["target"]

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        random_state=params["train"]["random_state"],
        max_iter=params["train"]["max_iter"],
        C=params["train"]["C"]
    ))
])

model.fit(X_train, y_train)

joblib.dump(model, "models/iris_model.joblib")

print("Model trained and saved successfully.")
