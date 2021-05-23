from fastapi.testclient import TestClient

from .app import app

client = TestClient(app)


def test_predict():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                    "age": 30,
                    "sex": 1,
                    "cp": 0,
                    "trestbps": 130,
                    "chol": 200,
                    "fbs": 0,
                    "restecg": 0,
                    "thalach": 150,
                    "exang": 0,
                    "oldpeak": 2,
                    "slope": 0,
                    "ca": 0,
                    "thal": 0
                    },
        )
        assert response.status_code == 200
        assert response.json() == {"prediction": 1}
