import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import click


def load_model(path: str) -> RandomForestClassifier:
    with open(path, "rb") as f:
        return pickle.load(f)


@click.command("validate_model")
@click.option("--data-dir")
@click.option("--model-dir")
@click.option("--metrics-dir")
def validate_model(data_dir:str, model_dir: str, metrics_dir:str):
    model = load_model(os.path.join(model_dir, "model.pkl"))

    val_data = pd.read_csv(os.path.join(data_dir, "val_data.csv"))

    y = val_data["target"].values
    X = val_data.drop("target", axis=1).values

    pred = model.predict(X)

    acc_score = accuracy_score(y, pred)

    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, "metrics.txt"), "w") as fin:
        fin.write(f"accuracy: {acc_score}")


if __name__ == '__main__':
    validate_model()