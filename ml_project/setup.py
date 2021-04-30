from setuptools import find_packages, setup

setup(
    name="heart_disease_classification",
    packages=find_packages(),
    version="0.1.0",
    description="Given physical attributes of a human,"
                "predict whether a heart disease is present in the patient or not",
    author="Shokhan Birlikov",
    install_requires=[
        "click==7.1.2",
        "python-dotenv>=0.5.1",
        "scikit-learn==0.24.1",
        "dataclasses==0.8",
        "pyyaml==3.11",
        "marshmallow-dataclass==8.3.0",
        "pandas==1.1.5",
    ],
    license="MIT",
)
