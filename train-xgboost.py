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
    n_estimators=1000,         # Começamos com um número alto de árvores
    learning_rate=0.05,        # Taxa de aprendizado
    n_jobs=-1,                 # Usa todos os núcleos da CPU
    random_state=42,
    early_stopping_rounds=50   # Para o treino se não houver melhora em 50 rodadas
)

# Faremos o mesmo para o DNI
xgb_model_dni = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    n_jobs=-1,
    random_state=42,
    early_stopping_rounds=50
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

# O resultado 'predictions' é um array numpy. Vamos convertê-lo para um DataFrame para facilitar a análise.
pred_df = pd.DataFrame({
    'ghi': predictions_ghi,
    'dni': predictions_dni
}, index=y_val.index)

print("\nAvaliando o desempenho do modelo...")

# Como temos duas saídas (GHI e DNI), vamos calcular o erro para cada uma.
# MAE (Mean Absolute Error): É o erro médio, na mesma unidade do alvo (W/m²). Fácil de interpretar.
# RMSE (Root Mean Squared Error): Penaliza erros maiores mais fortemente.

# Avaliação para GHI
mae_ghi = mean_absolute_error(y_val['ghi'], pred_df['ghi'])
rmse_ghi = np.sqrt(mean_squared_error(y_val['ghi'], pred_df['ghi']))

# Avaliação para DNI
mae_dni = mean_absolute_error(y_val['dni'], pred_df['dni'])
rmse_dni = np.sqrt(mean_squared_error(y_val['dni'], pred_df['dni']))

print("\n" + "="*50)
print("      RESULTADOS DE DESEMPENHO DO MODELO BASELINE (XGBoost)")
print("="*50)
print(f"\nTarget: GHI")
print(f"  - Erro Médio Absoluto (MAE): {mae_ghi:.2f} W/m²")
print(f"  - Raiz do Erro Quadrático Médio (RMSE): {rmse_ghi:.2f} W/m²")
print("-"*50)
print(f"Target: DNI")
print(f"  - Erro Médio Absoluto (MAE): {mae_dni:.2f} W/m²")
print(f"  - Raiz do Erro Quadrático Médio (RMSE): {rmse_dni:.2f} W/m²")
print("="*50)

# --- CONTEXTUALIZANDO O ERRO ---
# Para sabermos se um erro de X W/m² é bom ou ruim, vamos ver a média dos valores reais.
# Calculamos a média apenas durante o dia (quando a irradiação é > 0).
ghi_medio_dia = y_val[y_val['ghi'] > 0]['ghi'].mean()
dni_medio_dia = y_val[y_val['dni'] > 0]['dni'].mean()

print("\nPara Contexto:")
print(f"  - GHI médio (durante o dia) no set de validação: {ghi_medio_dia:.2f} W/m²")
print(f"  - DNI médio (durante o dia) no set de validação: {dni_medio_dia:.2f} W/m²")
print("="*50)

# Salvar o modelo treinado para uso futuro
print("\nSalvando o modelo XGBoost treinado...")
joblib.dump(xgb_model_ghi, 'training/xgb_model_ghi.joblib')
joblib.dump(xgb_model_dni, 'training/xgb_model_dni.joblib')
print("Modelos salvos como 'xgb_model_ghi.joblib' e 'xgb_model_dni.joblib'")