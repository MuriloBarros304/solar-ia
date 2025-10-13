from sklearn.ensemble import RandomForestRegressor
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

rf_model = RandomForestRegressor(
    n_estimators=100,      # Começaremos com 100 árvores. Um bom ponto de partida.
    n_jobs=-1,             # MUITO IMPORTANTE: Usa todos os núcleos da sua CPU para acelerar o treino.
    random_state=42,       # Garante que os resultados sejam reproduzíveis.
    verbose=2              # Mostra o progresso do treinamento árvore por árvore.
)

print("\nIniciando o treinamento do modelo... (Isso pode levar alguns minutos)")
start_time = time.time()

rf_model.fit(X_train, y_train)

end_time = time.time()
training_time = (end_time - start_time) / 60
print(f"Treinamento concluído em {training_time:.2f} minutos.")

print("\nRealizando previsões no conjunto de validação...")
predictions = rf_model.predict(X_val)

# O resultado 'predictions' é um array numpy. Vamos convertê-lo para um DataFrame para facilitar a análise.
pred_df = pd.DataFrame(predictions, index=y_val.index, columns=y_val.columns)

# Salvar o modelo treinado para uso futuro
print("\nSalvando o modelo RandomForest treinado...")
joblib.dump(rf_model, 'training/random_forest_model.joblib')
print("Modelo salvo como 'random_forest_model.joblib'")