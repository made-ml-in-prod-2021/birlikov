import pandas as pd
import pytest

from heart_disease_classification.data import read_data, split_train_val_data
from heart_disease_classification.params import SplittingParams


@pytest.fixture(scope='session')
def fake_dataset_csv_file(tmpdir_factory):
    N = 10
    index = range(N)
    even = [(n % 2 == 0) for n in range(N)]
    dataframe = pd.DataFrame({'even': even}, index=index)
    dataset_path = str(tmpdir_factory.mktemp('data').join('fake_dataset.csv'))
    dataframe.to_csv(dataset_path)
    return dataset_path


def test_load_dataset(dataset_path: str):
    data = read_data(dataset_path)
    assert isinstance(data, pd.DataFrame)
    assert data.shape[0] > 10


def test_split_dataset(tmpdir, dataset_path: str):
    val_size = 0.2
    splitting_params = SplittingParams(val_size=val_size, random_state=17)
    data = read_data(dataset_path)
    train_df, val_df = split_train_val_data(data, splitting_params)
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert train_df.shape[0] > val_df.shape[0] * (1 / val_size - 2)
    assert train_df.shape[0] < val_df.shape[0] * (1 / val_size)
