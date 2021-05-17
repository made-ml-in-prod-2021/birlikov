import pandas as pd
import pickle


from sklearn.pipeline import Pipeline
from pydantic import BaseModel, Field

from .custom_transformer_script import features


def load_object(path: str) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


class InputFeatures(BaseModel):
    age: int = Field(default=30, gt=0, description="Age in years.")
    sex: int = Field(default=1, description="1 - male, 0 - female.")
    cp: int = Field(default=0, description="Chest pain type: 0, 1, 2 or 3.")
    trestbps: int = Field(default=130, description="Resting blood pressure (in mm Hg on admission to the hospital).")
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


def make_predict(request: InputFeatures, model: Pipeline) -> Prediction:
    df = pd.DataFrame([dict(request)])
    pred = model.predict(df)
    return Prediction(prediction=pred)
