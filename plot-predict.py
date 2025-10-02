import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np

# --- 1. CONFIGURAÇÃO ---
print("Iniciando o script de visualização de previsões...")

# Caminhos para os modelos e dados
RF_MODEL_PATH = 'training/random_forest_model.joblib'
XGB_GHI_MODEL_PATH = 'training/xgb_model_ghi.joblib'
XGB_DNI_MODEL_PATH = 'training/xgb_model_dni.joblib'

X_VAL_PATH = 'data/X_val.parquet'
Y_VAL_PATH = 'data/y_val.parquet'

# <<<-- Escolha um período para visualizar -->>>
# Um período de 3 a 5 dias é ideal para ver os detalhes.
# Use o ano de 2023, que é o nosso conjunto de validação.
START_DATE = '2023-05-06'
END_DATE = '2023-05-07'

# --- 2. CARREGAR DADOS E MODELOS ---
print("Carregando dados e modelos...")
try:
    X_val = pd.read_parquet(X_VAL_PATH)
    y_val = pd.read_parquet(Y_VAL_PATH)
    
    rf_model = joblib.load(RF_MODEL_PATH)
    xgb_model_ghi = joblib.load(XGB_GHI_MODEL_PATH)
    xgb_model_dni = joblib.load(XGB_DNI_MODEL_PATH)
    print("Carregamento concluído.")
except FileNotFoundError as e:
    print(f"ERRO: Arquivo não encontrado: {e.filename}")
    print("Certifique-se de que os modelos foram treinados e salvos, e que os dados de validação existem.")
    exit()

# --- 3. GERAR PREVISÕES ---
print("Gerando previsões com os modelos carregados...")
# Previsões do RandomForest (multi-output)
pred_rf_raw = rf_model.predict(X_val)
pred_rf = pd.DataFrame(pred_rf_raw, index=y_val.index, columns=y_val.columns)

# Previsões do XGBoost (single-output)
pred_xgb_ghi = xgb_model_ghi.predict(X_val)
pred_xgb_dni = xgb_model_dni.predict(X_val)
pred_xgb = pd.DataFrame({'ghi': pred_xgb_ghi, 'dni': pred_xgb_dni}, index=y_val.index)

# --- 4. FILTRAR PERÍODO E PLOTAR ---
print(f"Filtrando dados para o período de {START_DATE} a {END_DATE}...")
y_val_period = y_val.loc[START_DATE:END_DATE]
pred_rf_period = pred_rf.loc[START_DATE:END_DATE]
pred_xgb_period = pred_xgb.loc[START_DATE:END_DATE]

print("Gerando gráficos...")
# Configura o estilo do gráfico
plt.style.use('seaborn-v0_8-whitegrid')
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(15, 10), sharex=True)

# --- Gráfico para GHI ---
axs[0].plot(y_val_period.index, y_val_period['ghi'], label='Valor Real', color='black', linewidth=2)
axs[0].plot(pred_rf_period.index, pred_rf_period['ghi'], label='RandomForest', color='blue', linestyle='--')
axs[0].plot(pred_xgb_period.index, pred_xgb_period['ghi'], label='XGBoost', color='red', linestyle=':')
axs[0].set_ylabel('GHI (W/m²)')
axs[0].set_title('Comparação de Previsões para GHI')
axs[0].legend()
axs[0].grid(True)

# --- Gráfico para DNI ---
axs[1].plot(y_val_period.index, y_val_period['dni'], label='Valor Real', color='black', linewidth=2)
axs[1].plot(pred_rf_period.index, pred_rf_period['dni'], label='RandomForest', color='blue', linestyle='--')
axs[1].plot(pred_xgb_period.index, pred_xgb_period['dni'], label='XGBoost', color='red', linestyle=':')
axs[1].set_xlabel('Data e Hora')
axs[1].set_ylabel('DNI (W/m²)')
axs[1].set_title('Comparação de Previsões para DNI')
axs[1].legend()
axs[1].grid(True)

# Melhora a formatação das datas no eixo X
fig.autofmt_xdate()
plt.tight_layout()
plt.show()