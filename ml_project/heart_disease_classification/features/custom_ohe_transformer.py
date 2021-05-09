import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class CustomOneHotTransformer(BaseEstimator):
    def __init__(self):
        self.columns = None
        self.ohe_columns = None

    def fit(self, df: pd.DataFrame):
        self.columns = df.columns.values

        ohe_cols = []
        for col in self.columns:
            ohe_cols.append(pd.get_dummies(df[col], prefix=col))

        self.ohe_columns = pd.concat(ohe_cols, axis=1).columns
        return self

    def transform(self, df):
        # check that we have a DataFrame with same column names as the one we fit
        if len(self.columns) != len(df.columns) or set(self.columns) != set(df.columns):
            raise ValueError('Passed DataFrame has different columns than fit DataFrame')

        ohe_df = pd.DataFrame(index=df.index, columns=self.ohe_columns)
        for ohe_col in ohe_df.columns:
            col, val = ohe_col.split("_")
            ohe_df[ohe_col] = df[col] == val

        return ohe_df