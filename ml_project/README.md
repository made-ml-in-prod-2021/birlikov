Machine Learning in Production - Homework-1

Author: Shokhan Birlikov

Dataset: https://www.kaggle.com/ronitf/heart-disease-uci

##### Setup:
```shell script
$ conda create --name ml_prod_hw1 python=3.6
$ conda activate ml_prod_hw1
$ pip install -r requirements.txt
```

##### Usage:
```shell script
$ python train_pipeline.py [OPTIONS] CONFIG_PATH
```

##### Example:
```shell script
$ python train_pipeline.py configs/train_config_2.yaml
```

##### Test:
```shell script
$ pytest tests/*
```

##### Prject overview:
```
├── LICENSE
├── README.md       <- Readme for ml_project
├── configs         <- training configs
├── data
│   └── raw         <- Original dataset
├── heart_disease_classification            <- Main folder with source codes
│   ├── __init__.py
│   ├── data                                <- Scripts for data preprocessing
│   ├── features                            <- Scripts for feature extraction
│   ├── models                              <- Scripts for model training and testing
│   └── params                              <- Scripts for parameter dataclasses
├── logger.py                               <- Logger
├── notebooks                               <- Notebooks with EDA
├── requirements.txt                        <- The requirements file for reproducing
├── saved_models                            <- Serialized trained models and metrics on validation set
├── setup.py                                <- Makes project pip installable
├── tests                                   <- Test for source codes
└── train_pipeline.py                       <- Main entry point script
```
