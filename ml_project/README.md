Machine Learning in Production - Homework-1

Author: Shokhan Birlikov

Dataset: https://www.kaggle.com/ronitf/heart-disease-uci

##### Setup
```python
$ conda create --name ml_prod_hw1 python=3.6
$ conda activate ml_prod_hw1
$ pip install -r requirements.txt
```

##### Usage
```python
$ python train_pipeline.py [OPTIONS] CONFIG_PATH
```

##### Example
```python
$ python train_pipeline.py configs/train_config_2.yaml
```