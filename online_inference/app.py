import os
from typing import Optional, Union, Tuple
import logging

import pandas as pd
import pickle
import uvicorn
from fastapi import FastAPI, HTTPException

from sklearn.pipeline import Pipeline
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)

model: Optional[Pipeline] = None

app = FastAPI()

EXPECTED_COLUMNS = ["age", "sex", "cp",
                    "trestbps", "chol", "fbs",
                    "restecg", "thalach", "exang",
                    "oldpeak", "slope", "ca", "thal"]


def load_object(path: str) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


class InputFeatures(BaseModel):
    age: int = Field(default=None, gt=0, description="Age in years.")
    sex: int = Field(default=0, lt=2, description="1 - male, 0 - female.")
    cp: int = Field(default=0, description="Chest pain type: 0, 1, 2 or 3.")
    trestbps: int = Field(default=120, description="Resting blood pressure (in mm Hg on admission to the hospital).")
    chol: int = Field(default=200, description="Serum cholestoral in mg/dl.")
    fbs: int = Field(default=0, description="(fasting blood sugar > 120 mg/dl) (1 = true; 0 = false).")
    restecg: int = Field(default=0, description="Resting electrocardiographic results.")
    thalach: int = Field(default=150, description="Maximum heart rate achieved.")
    exang: int = Field(default=0, description="Exercise induced angina (1 = yes; 0 = no).")
    oldpeak: float = Field(default=2.0, description="ST depression induced by exercise relative to rest.")
    slope: int = Field(default=0, description="The slope of the peak exercise ST segment.")
    ca: int = Field(default=0, description="Number of major vessels (0-3) colored by flourosopy.")
    thal: int = Field(default=0, description="Thalium Stress Test Result.")


class Prediction(BaseModel):
    prediction: int


def make_predict(request: InputFeatures, model: Pipeline) \
        -> Union[HTTPException, Prediction]:
    df = pd.DataFrame([dict(request)])

    # adding toy data validation
    if df['cp'].values[0] not in [0, 1, 2, 3]:
        raise HTTPException(status_code=400, detail="Chest pain type must be on of: 0, 1, 2 or 3.")

    pred = model.predict(df)
    return Prediction(prediction=pred)



@app.get("/")
def main():
    return "it is entry point of our predictor"


@app.on_event("startup")
def load_model():
    global model
    model_path = os.getenv("PATH_TO_MODEL", "RF_clf_pipeline.pkl")
    if model_path is None:
        err = f"PATH_TO_MODEL {model_path} is None"
        logger.error(err)
        raise RuntimeError(err)

    model = load_object(model_path)


@app.get("/health")
def health() -> bool:
    return not (model is None)


@app.post("/predict")
def predict(request: InputFeatures):
    return make_predict(request, model)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))