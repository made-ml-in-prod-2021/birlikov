### Запуск локально
```shell script
$ docker build -t shb/online_inference:v1 . 
$ docker run -p 8000:8000 shb/online_inference:v1
```
----------
### Запуск с докер-хаба
```shell script
$ docker pull shokhan/made_ml_prod_hw2_online_inference:latest
$ docker run -p 8000:8000 shokhan/made_ml_prod_hw2_online_inference
```
----------
#### Тестовый запрос:
```shell script
$ python make_request.py
```
----------
#### Оптимизация докер образа:
Изначально строил образ на `python:3.6` - получилось 1.29GB. Попробовал вместо этого `python:3.6-slim` - стало 524MB.