import pandas as pd

from heart_disease_classification.features import CustomOneHotTransformer


def test_custom_ohe_transformer():
    N = 10
    index = range(N)
    even = [(n % 2 == 0) for n in range(N)]
    odd = [(n % 2 == 1) for n in range(N)]
    cat_df = pd.DataFrame({'even': even, 'odd': odd}, index=index)

    custom_ohe_transformer = CustomOneHotTransformer()
    custom_ohe_transformer.fit(cat_df)

    assert len(custom_ohe_transformer.ohe_columns) == 4

    ohe_df = custom_ohe_transformer.transform(cat_df)

    assert isinstance(ohe_df, pd.DataFrame)
    assert (N, 4) == ohe_df.shape
