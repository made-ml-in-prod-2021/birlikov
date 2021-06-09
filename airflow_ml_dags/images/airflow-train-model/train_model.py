import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import click
import pickle


def serialize_model(model: object, output: str):
    with open(output, "wb") as f:
        pickle.dump(model, f)


@click.command("train_model")
@click.option("--data-dir")
@click.option("--model-dir")
def train_model(data_dir: str, model_dir):
    data = pd.read_csv(os.path.join(data_dir, "train_data.csv"))

    y = data["target"].values
    X = data.drop("target", axis=1).values

    model = RandomForestClassifier()

    model.fit(X, y)

    os.makedirs(model_dir, exist_ok=True)
    serialize_model(model, os.path.join(model_dir, "model.pkl"))


if __name__ == '__main__':
    train_model()