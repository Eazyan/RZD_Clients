
# Веб-приложение для предсказания оттока клиентов



Это веб-приложение, разработанное с использованием FastAPI и AutoGluon, для предсказания вероятности оттока клиентов на основе загружаемых данных.

  

## Установка

  

1. **Склонируйте репозиторий:**

  

```bash

git clone https://github.com/eazyan/RZD_Clients.git

cd RZD_Clients

```

  

2. **Создайте и активируйте виртуальное окружение с помощью Conda:**

  

```bash

conda create -n yourenv python=3.10

conda activate yourenv

```

  

3. **Установите зависимости:**

  

```bash

conda install --file requirements.txt

```

  

4. **Запустите приложение:**

  

```bash

uvicorn index:app --reload

```

  

Приложение будет доступно по адресу `http://127.0.0.1:8000`.

  

## Использование

  

1. Перейдите по адресу `http://127.0.0.1:8000` в вашем браузере.

2. Загрузите файл в формате CSV или Excel.

3. Получите предсказание вероятности оттока клиентов.

  

## Требования

- FastAPI

- Uvicorn

- AutoGluon

- Pandas

- Scikit-learn

Обратите внимание, что `AutoGluon` не добавлен в `requirements.txt`, так как его установка осуществляется через `conda`. AutoGluon установите через команду: conda install -c auto-gluon autogluon