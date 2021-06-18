import os

import click
from sklearn.datasets import load_iris


@click.command("generate_data")
@click.argument("output_dir")
def generate_data_function(output_dir: str):
    features, target = load_iris(return_X_y=True, as_frame=True)

    os.makedirs(output_dir, exist_ok=True)

    features.to_csv(os.path.join(output_dir, "features.csv"), index=False)
    target.to_csv(os.path.join(output_dir, "target.csv"), index=False)


if __name__ == '__main__':
    generate_data_function()