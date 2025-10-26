#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}


import json
import gzip
import pickle
import zipfile
import os
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import numpy as np # <-- Nuevo
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)


class Config:
    BASE_DIR = Path(__file__).parent.parent
    INPUT_DIR = BASE_DIR / "files" / "input"
    MODEL_DIR = BASE_DIR / "files" / "models"
    OUTPUT_DIR = BASE_DIR / "files" / "output"
    TRAIN_ZIP = INPUT_DIR / "train_data.csv.zip"
    TEST_ZIP = INPUT_DIR / "test_data.csv.zip"
    TRAIN_CSV_NAME = "train_data.csv"
    TEST_CSV_NAME = "test_data.csv"
    TARGET_COL = "Present_Price"
    CURRENT_YEAR = 2021
    CAT_COLS = ["Fuel_Type", "Selling_type", "Transmission"]
    SCALE_COLS = ["Driven_kms", "Owner", "Age"]
    PRICE_COLS = ["Selling_Price"] 
    PRICE_TARGET_COLS = ["Selling_Price", TARGET_COL] # Para transformación logarítmica
    
    # Hiperparámetros para GridSearchCV
    GRID_PARAMS = {
        "kbest__k": [4, 5, 6, 7, 8, 9, 10, 11], 
    }


def read_zipped_csv(zip_path: Path, csv_name: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(csv_name) as f:
            df = pd.read_csv(f)
            if df.columns[0].startswith('Unnamed'):
                return df.drop(columns=[df.columns[0]])
            return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned['Age'] = Config.CURRENT_YEAR - cleaned['Year']
    cleaned = cleaned.drop(['Year', 'Car_Name'], axis=1)
    cleaned = cleaned.dropna()
    
    # Aplicar transformación logarítmica a la feature de precio
    cleaned['Selling_Price'] = np.log1p(cleaned['Selling_Price'])
    
    return cleaned


def build_pipeline_search() -> GridSearchCV:
    
    preprocess = ColumnTransformer(
        transformers=[
            # 1. Transforma las variables categoricas usando one-hot-encoding.
            ("cat", OneHotEncoder(handle_unknown="ignore"), Config.CAT_COLS),
            # 2. Escala las variables numéricas al intervalo [0, 1].
            ("scale", MinMaxScaler(), Config.SCALE_COLS),
            # 3. La variable Selling_Price ya está transformada, la pasamos directamente
            ("price_pass", 'passthrough', Config.PRICE_COLS), 
        ],
        remainder="drop", 
    )
    
    pipe = Pipeline(
        steps=[
            ("prep", preprocess),
            # Selecciona las K mejores entradas.
            ("kbest", SelectKBest(score_func=f_regression)),
            # Ajusta un modelo de regresión lineal.
            ("linreg", LinearRegression()),
        ]
    )
    
    # Optimización de hiperparámetros con validación cruzada
    return GridSearchCV(
        estimator=pipe,
        param_grid=Config.GRID_PARAMS,
        cv=10,
        refit=True,
        verbose=1,
        # Usa el error medio absoluto negado para la optimización.
        scoring="neg_mean_absolute_error",
    )


def calculate_metrics(dataset_name: str, y_true, y_pred) -> Dict[str, Any]:
    return {
        "type": "metrics",
        "dataset": dataset_name,
        "r2": float(r2_score(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mad": float(mean_absolute_error(y_true, y_pred)),
    }


def save_model(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as fh:
        pickle.dump(model, fh)


def save_metrics_jsonl(metrics: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for metric in metrics:
            f.write(json.dumps(metric) + "\n")


def main():
    # Cargar y limpiar datasets 
    df_train = clean_dataset(read_zipped_csv(Config.TRAIN_ZIP, Config.TRAIN_CSV_NAME))
    df_test = clean_dataset(read_zipped_csv(Config.TEST_ZIP, Config.TEST_CSV_NAME))
    
    # Dividir en X e y 
    X_train = df_train.drop(Config.TARGET_COL, axis=1)
    # Aplicar transformación logarítmica a la variable objetivo
    y_train = np.log1p(df_train[Config.TARGET_COL])
    X_test = df_test.drop(Config.TARGET_COL, axis=1)
    y_test = np.log1p(df_test[Config.TARGET_COL])
    
    # Entrenar y optimizar modelo 
    search = build_pipeline_search()
    search.fit(X_train, y_train) 
    
    # Guardar modelo
    model_path = Config.MODEL_DIR / "model.pkl.gz"
    save_model(search, model_path)
    
    # Generar predicciones y calcular métricas
    y_train_pred_log = search.predict(X_train)
    y_test_pred_log = search.predict(X_test)
    
    # Revertir la transformación logarítmica para el cálculo de métricas
    y_train_pred = np.expm1(y_train_pred_log)
    y_test_pred = np.expm1(y_test_pred_log)
    
    # La variable y_true debe revertirse para el cálculo de métricas en la escala original.
    y_train_original = np.expm1(y_train)
    y_test_original = np.expm1(y_test)
    
    # Calcular métricas
    train_metrics = calculate_metrics("train", y_train_original, y_train_pred)
    test_metrics = calculate_metrics("test", y_test_original, y_test_pred)
    
    all_results = [train_metrics, test_metrics]
    
    # Guardar métricas
    metrics_path = Config.OUTPUT_DIR / "metrics.json"
    save_metrics_jsonl(all_results, metrics_path)


if __name__ == "__main__":
    main()