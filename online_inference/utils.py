import numpy as np
import pandas as pd
import pickle


from sklearn.pipeline import Pipeline
from pydantic import BaseModel, conlist
from typing import List, Union


def load_object(path: str) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


class InputRequest(BaseModel):
    data: str
    features: List[str]


class Prediction(BaseModel):
    prediction: int


def make_predict(request: InputRequest, model: Pipeline) -> Prediction:
    data = pd.DataFrame(request.data, columns=request.features)
    pred = model.predict(data)
    return Prediction(prediction=pred)