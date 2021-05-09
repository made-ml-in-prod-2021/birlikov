import os
import pickle
from sklearn.utils.validation import check_is_fitted
from typing import List, Tuple

import pandas as pd
import pytest
from py._path.local import LocalPath
from sklearn.ensemble import RandomForestClassifier

from heart_disease_classification.data.make_dataset import read_data
from heart_disease_classification.params import TrainingParams
from heart_disease_classification.params import FeatureParams
from heart_disease_classification.features.build_features import make_features, extract_target, build_transformer
from heart_disease_classification.models.model_fit_predict import train_model, serialize_model


@pytest.fixture(scope='session')
def fake_dataset_csv_file(tmpdir_factory):
    N = 100
    index = range(N)
    even = [str((n % 2 == 0)) for n in range(N)]
    odd = [(n % 2 == 1) for n in range(N)]
    target = [(n % 4 == 0) for n in range(N)]
    dataframe = pd.DataFrame(
        {"even": even,
         "odd": odd,
         "target": target
         },
        index=index)
    dataset_path = str(tmpdir_factory.mktemp('data').join('fake_dataset.csv'))
    dataframe.to_csv(dataset_path)
    return dataset_path


@pytest.fixture
def features_and_target(fake_dataset_csv_file: str) -> Tuple[pd.DataFrame, pd.Series]:
    params = FeatureParams(
        categorical_features=["even"],
        numerical_features=["odd"],
        features_to_drop=["ca"],
        target_col="target",
    )
    data = read_data(fake_dataset_csv_file)
    transformer = build_transformer(params)
    transformer.fit(data)
    features = make_features(transformer, data, params)
    target = extract_target(data, params)
    return features, target


def test_train_model(features_and_target: Tuple[pd.DataFrame, pd.Series]):
    features, target = features_and_target
    model = train_model(features, target, train_params=TrainingParams())
    assert isinstance(model, RandomForestClassifier)
    check_is_fitted(model)


def test_serialize_model(tmpdir: LocalPath):
    expected_output = tmpdir.join("clf_model.pkl")
    n_estimators = 10
    model = RandomForestClassifier(n_estimators=n_estimators)
    real_output = serialize_model(model, expected_output)
    assert real_output == expected_output
    assert os.path.exists
    with open(real_output, "rb") as f:
        model = pickle.load(f)
    assert isinstance(model, RandomForestClassifier)