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
        "py==1.10.0",
        "scikit-learn==0.24.2",
        "dataclasses==0.8",
        "pyyaml==5.4.1",
        "marshmallow-dataclass==8.4.1",
        "pandas==1.1.5",
        "numpy==1.19.5",
        "pytest==6.2.4",
    ],
    license="MIT",
)
