import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

df = pd.read_csv("data/iris.csv")

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=params["prepare"]["test_size"],
    random_state=params["prepare"]["random_state"],
    stratify=y
)

train_df = X_train.copy()
train_df["target"] = y_train

test_df = X_test.copy()
test_df["target"] = y_test

train_df.to_csv("data/processed/train.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

print("Training and testing datasets created successfully.")
