import requests

if __name__ == "__main__":
    json_data = {
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
                    }

    response = requests.post(
        "http://localhost:52376/predict",
        json=json_data,
    )
    print(f"Status code: {response.status_code}")
    print(f"Response body: {response.json()}")
