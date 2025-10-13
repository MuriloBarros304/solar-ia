import pandas as pd
import matplotlib.pyplot as plt
import joblib

# --- 1. CARREGAR DADOS E MODELOS (IGUAL AO SCRIPT ANTERIOR) ---
print("Carregando dados de validação e modelos treinados...")
try:
    X_val = pd.read_parquet('data/X_val.parquet')
    y_val = pd.read_parquet('data/y_val.parquet')
    
    rf_model = joblib.load('training/random_forest_model.joblib')
    xgb_model_ghi = joblib.load('training/xgb_model_ghi.joblib')
    xgb_model_dni = joblib.load('training/xgb_model_dni.joblib')
    print("Carregamento concluído.")
except FileNotFoundError as e:
    print(f"ERRO: Arquivo não encontrado: {e.filename}")
    print("Certifique-se de que os modelos foram treinados e salvos.")
    exit()

# --- 2. GERAR PREVISÕES PARA O ANO INTEIRO ---
print("Gerando previsões para todo o conjunto de validação...")
pred_rf_raw = rf_model.predict(X_val)
pred_rf = pd.DataFrame(pred_rf_raw, index=y_val.index, columns=y_val.columns)

pred_xgb_ghi = xgb_model_ghi.predict(X_val)
pred_xgb_dni = xgb_model_dni.predict(X_val)
pred_xgb = pd.DataFrame({'ghi': pred_xgb_ghi, 'dni': pred_xgb_dni}, index=y_val.index)
print("Previsões geradas.")

# --- 3. AGREGAR DADOS HORÁRIOS PARA DIÁRIOS ---
print("Agregando dados de hora em hora para uma resolução diária...")

# Calculando a MÉDIA diária
y_val_daily_mean = y_val.resample('D').mean()
pred_rf_daily_mean = pred_rf.resample('D').mean()
pred_xgb_daily_mean = pred_xgb.resample('D').mean()

# Calculando a ENERGIA TOTAL diária (soma das potências horárias) e convertendo para kWh/m²
y_val_daily_kwh = y_val.resample('D').sum() / 1000
pred_rf_daily_kwh = pred_rf.resample('D').sum() / 1000
pred_xgb_daily_kwh = pred_xgb.resample('D').sum() / 1000

# --- 4. PLOTAR GRÁFICOS ANUAIS ---
print("Gerando gráficos de análise anual...")
plt.style.use('seaborn-v0_8-whitegrid')

# ----- GRÁFICO 1: MÉDIA DIÁRIA DE IRRADIÂNCIA -----
fig1, axs1 = plt.subplots(nrows=2, ncols=1, figsize=(15, 10), sharex=True)
fig1.suptitle('Análise Anual: Irradiância Média Diária (W/m²)', fontsize=16)

# GHI Média Diária
axs1[0].plot(y_val_daily_mean.index, y_val_daily_mean['ghi'], label='Real', color='black', alpha=0.7)
axs1[0].plot(pred_rf_daily_mean.index, pred_rf_daily_mean['ghi'], label='RandomForest', color='blue', linestyle='--', alpha=0.8)
axs1[0].plot(pred_xgb_daily_mean.index, pred_xgb_daily_mean['ghi'], label='XGBoost', color='red', linestyle=':', alpha=0.8)
axs1[0].set_ylabel('GHI Médio (W/m²)')
axs1[0].legend()

# DNI Média Diária
axs1[1].plot(y_val_daily_mean.index, y_val_daily_mean['dni'], label='Real', color='black', alpha=0.7)
axs1[1].plot(pred_rf_daily_mean.index, pred_rf_daily_mean['dni'], label='RandomForest', color='blue', linestyle='--', alpha=0.8)
axs1[1].plot(pred_xgb_daily_mean.index, pred_xgb_daily_mean['dni'], label='XGBoost', color='red', linestyle=':', alpha=0.8)
axs1[1].set_xlabel('Data (Ano de 2023)')
axs1[1].set_ylabel('DNI Médio (W/m²)')
axs1[1].legend()

fig1.autofmt_xdate()
plt.tight_layout(rect=(0, 0.03, 1, 0.95)) # Ajusta para o supertítulo

# ----- GRÁFICO 2: INSOLAÇÃO TOTAL DIÁRIA -----
fig2, axs2 = plt.subplots(nrows=2, ncols=1, figsize=(15, 10), sharex=True)
fig2.suptitle('Análise Anual: Insolação Total Diária (kWh/m²/dia)', fontsize=16)

# GHI Total Diário
axs2[0].plot(y_val_daily_kwh.index, y_val_daily_kwh['ghi'], label='Real', color='black', alpha=0.7)
axs2[0].plot(pred_rf_daily_kwh.index, pred_rf_daily_kwh['ghi'], label='RandomForest', color='blue', linestyle='--', alpha=0.8)
axs2[0].plot(pred_xgb_daily_kwh.index, pred_xgb_daily_kwh['ghi'], label='XGBoost', color='red', linestyle=':', alpha=0.8)
axs2[0].set_ylabel('Insolação GHI (kWh/m²)')
axs2[0].legend()

# DNI Total Diário
axs2[1].plot(y_val_daily_kwh.index, y_val_daily_kwh['dni'], label='Real', color='black', alpha=0.7)
axs2[1].plot(pred_rf_daily_kwh.index, pred_rf_daily_kwh['dni'], label='RandomForest', color='blue', linestyle='--', alpha=0.8)
axs2[1].plot(pred_xgb_daily_kwh.index, pred_xgb_daily_kwh['dni'], label='XGBoost', color='red', linestyle=':', alpha=0.8)
axs2[1].set_xlabel('Data (Ano de 2023)')
axs2[1].set_ylabel('Insolação DNI (kWh/m²)')
axs2[1].legend()

fig2.autofmt_xdate()
plt.tight_layout(rect=(0, 0.03, 1, 0.95))

# (Dentro da seção de plotagem do GRÁFICO 2)

ROLLING_WINDOW = 30 # Média móvel de 30 dias

# Adicione esta linha na plotagem do GHI
axs2[0].plot(y_val_daily_kwh.index, y_val_daily_kwh['ghi'].rolling(window=ROLLING_WINDOW).mean(), label=f'Tendência ({ROLLING_WINDOW} dias)', color='green', linewidth=3)

# Adicione esta linha na plotagem do DNI
axs2[1].plot(y_val_daily_kwh.index, y_val_daily_kwh['dni'].rolling(window=ROLLING_WINDOW).mean(), label=f'Tendência ({ROLLING_WINDOW} dias)', color='green', linewidth=3)
# E não se esqueça de adicionar a legenda novamente para que a nova linha apareça
axs2[0].legend()
axs2[1].legend()

plt.show()