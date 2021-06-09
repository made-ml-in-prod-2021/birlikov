import os
import pandas as pd
import click
from sklearn.model_selection import train_test_split


@click.command("split_data")
@click.option("--data-dir")
def split_data(data_dir: str):
    data = pd.read_csv(os.path.join(data_dir, "preprocessed_data.csv"))

    train_data, val_data = train_test_split(data, test_size=0.2)

    train_data.to_csv(os.path.join(data_dir, "train_data.csv"), index=False)
    val_data.to_csv(os.path.join(data_dir, "val_data.csv"), index=False)


if __name__ == '__main__':
    split_data()