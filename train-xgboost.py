import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
import time
import joblib

print("Carregando os conjuntos de treino e validação...")
try:
    X_train = pd.read_parquet('data/X_train.parquet')
    y_train = pd.read_parquet('data/y_train.parquet')
    X_val = pd.read_parquet('data/X_val.parquet')
    y_val = pd.read_parquet('data/y_val.parquet')
    print("Dados carregados com sucesso.")
except FileNotFoundError:
    print("ERRO: Arquivos de treino/validação não encontrados. Execute o script de separação de dados primeiro.")
    exit()

# XGBoost pode treinar um modelo para cada alvo separadamente.
xgb_model_ghi = xgb.XGBRegressor(
    n_estimators=800,          # Começamos com um número alto de árvores
    learning_rate=0.05,        # Taxa de aprendizado
    n_jobs=-1,                 # Usa todos os núcleos da CPU
    random_state=42,
    early_stopping_rounds=50,  # Para o treino se não houver melhora em 50 rodadas
    max_depth=9,               # Profundidade máxima das árvores
    subsample=0.9,             # Amostra 90% dos dados para cada árvore
    colsample_bytree=0.93      # Amostra 80% das features para cada árvore
)

# Faremos o mesmo para o DNI
xgb_model_dni = xgb.XGBRegressor(
    n_estimators=800,          # Começamos com um número alto de árvores
    learning_rate=0.05,        # Taxa de aprendizado
    n_jobs=-1,                 # Usa todos os núcleos da CPU
    random_state=42,
    early_stopping_rounds=50,  # Para o treino se não houver melhora em 50 rodadas
    max_depth=9,               # Profundidade máxima das árvores
    subsample=0.9,             # Amostra 90% dos dados para cada árvore
    colsample_bytree=0.93      # Amostra 80% das features para cada árvore
)

print("\nIniciando o treinamento do modelo... (Isso pode levar alguns minutos)")
start_time = time.time()

xgb_model_ghi.fit(X_train, y_train['ghi'], eval_set=[(X_val, y_val['ghi'])], verbose=100)
xgb_model_dni.fit(X_train, y_train['dni'], eval_set=[(X_val, y_val['dni'])], verbose=100)

end_time = time.time()
training_time = (end_time - start_time) / 60
print(f"Treinamento concluído em {training_time:.2f} minutos.")

print("\nRealizando previsões no conjunto de validação...")
predictions_ghi = xgb_model_ghi.predict(X_val)
predictions_dni = xgb_model_dni.predict(X_val)

# O resultado 'predictions' é um array numpy. Convertê-lo para um DataFrame para facilitar a análise.
pred_df = pd.DataFrame({
    'ghi': predictions_ghi,
    'dni': predictions_dni
}, index=y_val.index)

# Salvar o modelo treinado para uso futuro
print("\nSalvando o modelo XGBoost treinado...")
joblib.dump(xgb_model_ghi, 'training/xgb_model_ghi.joblib')
joblib.dump(xgb_model_dni, 'training/xgb_model_dni.joblib')
print("Modelos salvos como 'xgb_model_ghi.joblib' e 'xgb_model_dni.joblib'")