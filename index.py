from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from autogluon.tabular import TabularPredictor, TabularDataset
import pandas as pd
from io import BytesIO
import os
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

# Загружаем предобученный predictor AutoGluon
predictor_path = "AutogluonModels/ag-20241027_042736"  # Путь к модели
if not os.path.exists(predictor_path):
    raise FileNotFoundError(f"Модель не найдена по пути {predictor_path}")
predictor = TabularPredictor.load(predictor_path)

# Инициализация объектов
label_encoder = LabelEncoder()

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        'ID', 'Тема вопроса', 'Находится в реестре МСП', 'Размер компании.Наименование',
        'Размер уставного капитала объявленный', 'ОКВЭД2.Код', 'ЕЛС действующий',
        'Грузоотправитель', 'Грузополучатель', 
        'Карточка клиента (внешний источник).Индекс платежной дисциплины Значение', 
        'Карточка клиента (внешний источник).Индекс финансового риска Значение', 
        'Госконтракты.Тип контракта', 'Сценарий', 'Ожидаемая выручка', 
        'Вероятность сделки, %', 'Канал первичного интереса', 'Состояние', 'Код груза'
    ]

    return df

async def convert_to_dataframe(file: UploadFile) -> pd.DataFrame:
    """Конвертер для обработки различных табличных форматов и возвращения DataFrame."""
    try:
        contents = await file.read()
        if file.filename.endswith('.csv'):
            df = pd.read_csv(BytesIO(contents))
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Формат файла не поддерживается. Пожалуйста, загрузите CSV или Excel файл.")
        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке файла: {str(e)}")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Конвертация файла в DataFrame
    df = await convert_to_dataframe(file)
    
    # Препроцессинг данных
    df = preprocess_data(df)

    # Проверяем наличие целевой колонки и удаляем её, если она присутствует
    if 'Target' in df.columns:
        df = df.drop(columns=['Target'])

    # Преобразуем данные в формат, поддерживаемый AutoGluon
    test_data = TabularDataset(df)

    # Выполняем предсказание
    y_pred = predictor.predict(test_data)

    # Преобразуем предсказания в проценты с проверкой диапазона
    y_pred_percent = [min(max(int(value * 100), 0), 100) for value in y_pred]

    html_table = """
    <table>
        <tr><th>Индекс</th><th>Вероятность оттока (%)</th></tr>
        {}
    </table>
    """.format(''.join(f"<tr><td>{i + 1}</td><td>{value}%</td></tr>" for i, value in enumerate(y_pred_percent)))

    return HTMLResponse(content=html_table)


@app.get("/", response_class=HTMLResponse)
async def main():
    html_content = """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Предсказание оттока клиентов</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                background-color: #f0f4f8;
            }
            .container {
                background-color: #ffffff;
                padding: 2em;
                border-radius: 8px;
                box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                width: 100%;
                max-width: 600px;
                text-align: center;
            }
            h3 {
                color: #333333;
            }
            input[type="file"] {
                display: none;
            }
            label {
                display: inline-block;
                padding: 1em 2em;
                margin: 1em 0;
                color: white;
                background-color: #007BFF;
                border-radius: 4px;
                cursor: pointer;
            }
            label:hover {
                background-color: #0056b3;
            }
            button {
                display: inline-block;
                padding: 1em 2em;
                margin: 1em 0;
                color: white;
                background-color: #007BFF;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 1em;
            }
            button:hover {
                background-color: #0056b3;
            }
            .result {
                margin-top: 1em;
                display: none;
                max-height: 300px; /* Ограничиваем высоту */
                overflow-y: auto; /* Добавляем прокрутку */
                border: 1px solid #dddddd;
                border-radius: 4px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
            }
            th, td {
                padding: 0.75em;
                text-align: center;
                border: 1px solid #dddddd;
            }
            th {
                background-color: #007BFF;
                color: white;
            }
            .error {
                color: #FF0000;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h3>Загрузите файл для предсказания оттока</h3>
            <form id="upload-form" method="post" enctype="multipart/form-data">
                <label for="file-upload">Выберете файл</label>
                <input id="file-upload" name="file" type="file" accept=".csv, .xls, .xlsx" required>
                <button type="submit">Отправить</button>
            </form>
            <div class="result" id="result"></div>
            <div class="error" id="error"></div>
        </div>
        
        <script>
            const form = document.getElementById('upload-form');
            const fileInput = document.getElementById('file-upload');
            const resultDiv = document.getElementById('result');
            const errorDiv = document.getElementById('error');

            form.addEventListener('submit', async (event) => {
                event.preventDefault();
                
                const formData = new FormData();
                formData.append("file", fileInput.files[0]);

                resultDiv.style.display = "none";
                errorDiv.textContent = "";

                try {
                    const response = await fetch('/predict/', {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        const errorData = await response.json();
                        errorDiv.textContent = errorData.detail;
                        return;
                    }

                    const data = await response.text();
                    resultDiv.innerHTML = data; // Устанавливаем HTML из ответа
                    resultDiv.style.display = "block";
                } catch (error) {
                    errorDiv.textContent = "Ошибка: " + error.message;
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
