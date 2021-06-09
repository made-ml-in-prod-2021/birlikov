import os
import pandas as pd
import click


@click.command("preprocess_data")
@click.option("--input-dir")
@click.option("--output-dir")
def preprocess_data(input_dir: str, output_dir):
    data = pd.read_csv(os.path.join(input_dir, "features.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target.csv"))

    data["target"] = target["target"]

    data.dropna(inplace=True)

    os.makedirs(output_dir, exist_ok=True)
    data.to_csv(os.path.join(output_dir, "preprocessed_data.csv"), index=False)


if __name__ == '__main__':
    preprocess_data()