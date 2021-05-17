import os
from typing import Optional

import uvicorn
from fastapi import FastAPI
from sklearn.pipeline import Pipeline


from .logger import logger
from .utils import load_object, make_predict, InputFeatures, Prediction


model: Optional[Pipeline] = None

app = FastAPI()


@app.get("/")
def main():
    return "it is entry point of our predictor"


@app.on_event("startup")
def load_model():
    global model
    model_path = os.getenv("PATH_TO_MODEL", "RF_classifier.pkl")
    if model_path is None:
        err = f"PATH_TO_MODEL {model_path} is None"
        logger.error(err)
        raise RuntimeError(err)

    model = load_object(model_path)


@app.get("/health")
def health() -> bool:
    return not (model is None)


@app.get("/predict/", response_model=Prediction)
def predict(request: InputFeatures):
    return make_predict(request, model)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))