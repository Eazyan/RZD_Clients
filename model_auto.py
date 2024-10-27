import pandas as pd
from autogluon.tabular import TabularPredictor

# Шаг 1: Загрузка данных
file_path = 'fin_data.csv'
data = pd.read_csv(file_path)

# Шаг 2: Предварительная обработка данных
data = data.dropna(axis=1, thresh=len(data) * 0.5)
numeric_cols = data.select_dtypes(include='number').columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

categorical_cols = data.select_dtypes(include='object').columns
for col in categorical_cols:
    mode_value = data[col].mode()
    data[col] = data[col].fillna(mode_value.iloc[0] if not mode_value.empty else 'Unknown')

# Шаг 3: Выбор целевой переменной
target_column = 'Target'
if target_column not in data.columns:
    raise ValueError(f"Целевая переменная '{target_column}' не найдена в данных")

# Шаг 4: Разделение данных на обучающие и тестовые
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)

# Шаг 5: Обучение модели с помощью AutoGluon
predictor = TabularPredictor(label=target_column, eval_metric='r2').fit(train_data)

