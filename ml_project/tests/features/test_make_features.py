import pandas as pd
import pytest
from numpy.testing import assert_allclose

from heart_disease_classification.params.feature_params import FeatureParams
from heart_disease_classification.features.build_features import make_features, extract_target, build_transformer


@pytest.fixture
def toy_dataframe():
    return pd.DataFrame(
        {"categorical_feature": ["cat", "dog", "cow"],
         "numerical_feature": [1, 2, 3],
         "target_col": [1, 0, 1]}
    )


@pytest.fixture
def feature_params() -> FeatureParams:
    params = FeatureParams(
        categorical_features=["categorical_feature"],
        numerical_features=["numerical_feature"],
        features_to_drop=["some_other_col"],
        target_col="target_col",
    )
    return params


def test_make_features(toy_dataframe: pd.DataFrame, feature_params: FeatureParams):
    data = toy_dataframe
    transformer = build_transformer(feature_params)
    transformer.fit(data)
    features = make_features(transformer, data)
    assert not pd.isnull(features).any().any()


def test_extract_features(toy_dataframe: pd.DataFrame, feature_params: FeatureParams):
    data = toy_dataframe
    target = extract_target(data, feature_params)
    assert_allclose(
        data[feature_params.target_col].to_numpy(), target.to_numpy()
    )
