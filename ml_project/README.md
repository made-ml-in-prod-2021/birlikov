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
