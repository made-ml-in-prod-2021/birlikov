import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
import click

model_path = os.environ['MODEL_PATH']


def load_model(path: str) -> RandomForestClassifier:
    with open(path, "rb") as f:
        return pickle.load(f)


@click.command("get_predictions")
@click.option("--data-dir")
@click.option("--predictions-dir")
def get_predictions(data_dir: str, predictions_dir: str):

    features = pd.read_csv(os.path.join(data_dir, "features.csv"))
    features.fillna(0, inplace=True)

    print(f"\nUsing model_path: '{model_path}'\n")
    model = load_model(model_path)

    predictions = model.predict(features)

    features["predictions"] = predictions

    os.makedirs(predictions_dir, exist_ok=True)
    features.to_csv(os.path.join(predictions_dir, "predictions.csv"), index=False)


if __name__ == '__main__':
    get_predictions()