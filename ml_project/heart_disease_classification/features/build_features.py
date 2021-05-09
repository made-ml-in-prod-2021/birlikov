import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from heart_disease_classification.params import FeatureParams

from heart_disease_classification.features.custom_ohe_transformer import CustomOneHotTransformer


def process_categorical_features(categorical_df: pd.DataFrame) -> pd.DataFrame:

    categorical_pipeline = build_categorical_pipeline()
    return categorical_pipeline.fit_transform(categorical_df)


def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        [
            ("custom_ohe", CustomOneHotTransformer()),
        ]
    )
    return categorical_pipeline


def process_numerical_features(numerical_df: pd.DataFrame) -> pd.DataFrame:
    num_pipeline = build_numerical_pipeline()
    return pd.DataFrame(num_pipeline.fit_transform(numerical_df), columns=numerical_df.columns)


def build_numerical_pipeline() -> Pipeline:
    num_pipeline = Pipeline(
        [("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),]
    )
    return num_pipeline


def make_features(transformer: ColumnTransformer, df: pd.DataFrame, params: FeatureParams) -> pd.DataFrame:
    cols_to_drop = [c for c in params.features_to_drop if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    transformed = transformer.transform(df)
    return transformed


def build_transformer(params: FeatureParams) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                "categorical_pipeline",
                build_categorical_pipeline(),
                params.categorical_features,
            ),
            (
                "numerical_pipeline",
                build_numerical_pipeline(),
                params.numerical_features,
            ),
        ]
    )
    return transformer


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    target = df[params.target_col]
    return target
