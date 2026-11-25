from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

BASE = Path(__file__).resolve().parent
URL = 'https://raw.githubusercontent.com/dankicode/Ames_Housing/main/AmesHousing.csv'
COLUNAS = ['Gr Liv Area', 'Bedroom AbvGr', 'Full Bath', 'Year Built', 'Overall Qual', 'Garage Cars']


def treinar():
    """Carrega a base Ames Housing, compara modelos na validação e salva o vencedor."""
    arquivo = BASE / 'dataset.csv'
    if not arquivo.exists():
        pd.read_csv(URL).to_csv(arquivo, index=False)

    df = pd.read_csv(arquivo).dropna(subset=['SalePrice'])
    X, y = df[COLUNAS], df['SalePrice']

    # Separação em 60% treino, 20% validação e 20% teste
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.25, random_state=42)

    modelos = {
        'Regressão linear': LinearRegression(),
        'Random Forest': RandomForestRegressor(
            n_estimators=120,
            random_state=42,
            min_samples_leaf=2,
            n_jobs=-1,
        ),
    }

    resultados = {}
    pipelines = {}
    for nome, algoritmo in modelos.items():
        modelo = make_pipeline(SimpleImputer(strategy='median'), algoritmo)
        modelo.fit(X_train, y_train)
        resultados[nome] = float(mean_absolute_error(y_val, modelo.predict(X_val)))
        pipelines[nome] = modelo

    melhor = min(resultados, key=resultados.get)
    modelo_final = pipelines[melhor].fit(X_dev, y_dev)
    pred = modelo_final.predict(X_test)

    metricas = {
        'registros': int(len(df)),
        'modelo': melhor,
        'mae_validacao': resultados,
        'mae': float(mean_absolute_error(y_test, pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, pred))),
        'r2': float(r2_score(y_test, pred)),
    }

    joblib.dump(modelo_final, BASE / 'modelo.joblib')
    (BASE / 'metricas.json').write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    return metricas


if __name__ == '__main__':
    print(treinar())
