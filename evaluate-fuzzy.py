"""
Módulo para avaliação e comparação dos sistemas fuzzy (Mamdani e Sugeno)
com as predições de Machine Learning (XGBoost).
Métrica unificada: W/m² (Irradiação Global Horizontal)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

from Fuzzy.ghi_mamdani import (
    avaliar_ghi_mamdani, 
    hora_sin, hora_cos, tipo_nuvem, temp_ar, ghi, controle_ghi # Adicionado controle_ghi para extrair regras
)
from Fuzzy.ghi_sugeno import avaliar_ghi_sugeno

# Configuração de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

OUTPUT_DIR = 'predict/fuzzy_evaluation'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("AVALIAÇÃO DE SISTEMAS HÍBRIDOS: ML vs FUZZY (GHI W/m²)")
print("=" * 70)

# ======================================================
# 1. CARREGAMENTO DOS DADOS
# ======================================================
print("\n[1/8] Carregando dados e modelos...")
X_TEST_PATH = 'data/X_test.parquet'
Y_TEST_PATH = 'data/y_test.parquet'

try:
    X_test = pd.read_parquet(X_TEST_PATH)
    y_test = pd.read_parquet(Y_TEST_PATH)
    model_ghi = joblib.load('training/xgb_model_ghi.joblib')
    features = joblib.load('training/model_features.joblib')
    print(f"Dados: {len(X_test)} amostras")
except FileNotFoundError:
    print("Erro: Arquivos de dados não encontrados.")
    exit(1)

# ======================================================
# 2. PREDIÇÕES ML
# ======================================================
print("\n[2/8] Gerando predições ML (XGBoost)...")
y_pred_xgb = model_ghi.predict(X_test[features])

# ======================================================
# 3. FILTRAGEM E ALINHAMENTO
# ======================================================
print("\n[3/8] Preparando amostra de teste...")
mask_unique = ~X_test.index.duplicated(keep='first')
X_final = X_test[mask_unique]
y_final = y_test[mask_unique]
y_pred_xgb_final = y_pred_xgb[mask_unique]

SAMPLE_SIZE = 1000 
X_sample = X_final.iloc[:SAMPLE_SIZE]
y_sample = y_final.iloc[:SAMPLE_SIZE]
y_xgb_sample = y_pred_xgb_final[:SAMPLE_SIZE]

print(f"✓ Amostra definida: {len(X_sample)} pontos")

# ======================================================
# 4. EXECUÇÃO DOS SISTEMAS FUZZY
# ======================================================
print("\n[4/8] Calculando inferência Fuzzy...")

mamdani_preds = []
sugeno_preds = []

for idx, row in X_sample.iterrows():
    h_sin = row['hora_sin']
    h_cos = row['hora_cos']
    nuvem = row['tipo_nuvem']
    temp = row['temp_ar']
    
    try:
        res_m = avaliar_ghi_mamdani(h_sin, h_cos, nuvem, temp)
        mamdani_preds.append(res_m)
    except:
        mamdani_preds.append(0.0)
        
    try:
        res_s = avaliar_ghi_sugeno(h_sin, h_cos, nuvem, temp)
        sugeno_preds.append(res_s)
    except:
        sugeno_preds.append(0.0)

df_eval = pd.DataFrame(index=X_sample.index)
df_eval['GHI_Real'] = y_sample['ghi']
df_eval['ML_XGBoost'] = y_xgb_sample
df_eval['Fuzzy_Mamdani'] = mamdani_preds
df_eval['Fuzzy_Sugeno'] = sugeno_preds
df_eval['Tipo_Nuvem'] = X_sample['tipo_nuvem']

# ======================================================
# 5. CÁLCULO DE MÉTRICAS
# ======================================================
print("\n[5/8] Calculando métricas de erro (Apenas Diurno)...")
df_diurno = df_eval[df_eval['GHI_Real'] > 10].copy()

metricas_log = []

def calc_metrics(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    msg = f"--- {name} (Diurno) ---\nMAE:  {mae:.2f} W/m²\nRMSE: {rmse:.2f} W/m²\nR²:   {r2:.4f}\n"
    print(msg)
    metricas_log.append(msg)
    
    return mae, rmse, r2

mae_ml, rmse_ml, r2_ml = calc_metrics(df_diurno['GHI_Real'], df_diurno['ML_XGBoost'], "XGBoost")
mae_mam, rmse_mam, r2_mam = calc_metrics(df_diurno['GHI_Real'], df_diurno['Fuzzy_Mamdani'], "Mamdani")
mae_sug, rmse_sug, r2_sug = calc_metrics(df_diurno['GHI_Real'], df_diurno['Fuzzy_Sugeno'], "Sugeno")

# ======================================================
# 6. GERAÇÃO DE GRÁFICOS
# ======================================================
print("\n[6/8] Gerando gráficos comparativos...")

# Série Temporal
plt.figure(figsize=(14, 6))
subset = df_eval.iloc[50:150]
plt.plot(subset.index, subset['GHI_Real'], 'k-', linewidth=2, label='Real', alpha=0.8)
plt.plot(subset.index, subset['ML_XGBoost'], 'b--', label='XGBoost', alpha=0.7)
plt.plot(subset.index, subset['Fuzzy_Mamdani'], 'orange', label='Mamdani', alpha=0.8)
plt.plot(subset.index, subset['Fuzzy_Sugeno'], 'r:', linewidth=2, label='Sugeno', alpha=0.8)
plt.title('Comparação Temporal (Zoom)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f'{OUTPUT_DIR}/series_temporal_ghi.png', dpi=300)
plt.close()

# Barras Métricas
plt.figure(figsize=(10, 6))
modelos = ['XGBoost', 'Mamdani', 'Sugeno']
maes = [mae_ml, mae_mam, mae_sug]
rmses = [rmse_ml, rmse_mam, rmse_sug]
x = np.arange(len(modelos))
w = 0.35
plt.bar(x - w/2, maes, w, label='MAE', color=['#90CAF9', '#FFCC80', '#EF9A9A'])
plt.bar(x + w/2, rmses, w, label='RMSE', color=['#1976D2', '#F57C00', '#D32F2F'])
plt.xticks(x, modelos)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.savefig(f'{OUTPUT_DIR}/barras_metricas.png', dpi=300)
plt.close()

# ======================================================
# 7. SALVAMENTO INDIVIDUAL DE IMAGENS E RELATÓRIO
# ======================================================
print("\n[7/8] Salvando funções de pertinência individuais...")

# Lista de variáveis para plotar
variaveis_fuzzy = [
    (hora_sin, 'Input - Ciclo Diário (Seno)', 'pertinencia_hora_sin.png'),
    (hora_cos, 'Input - Dia_Noite (Cosseno)', 'pertinencia_hora_cos.png'),
    (tipo_nuvem, 'Input - Cobertura de Nuvens', 'pertinencia_nuvens.png'),
    (temp_ar, 'Input - Temperatura do Ar', 'pertinencia_temp.png'),
    (ghi, 'Output - GHI (W_m2)', 'pertinencia_ghi_output.png')
]

for var, titulo, filename in variaveis_fuzzy:
    try:
        var.view()
        # Captura a figura atual gerada pelo skfuzzy
        plt.title(titulo)
        plt.savefig(f'{OUTPUT_DIR}/{filename}', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  -> Salvo: {filename}")
    except Exception as e:
        print(f"  -> Erro ao salvar {filename}: {e}")

# ======================================================
# 8. GERAR RELATÓRIO TXT
# ======================================================
print("\n[8/8] Gerando relatório de texto (Métricas e Regras)...")

txt_filename = f'{OUTPUT_DIR}/relatorio_metricas_regras.txt'
with open(txt_filename, 'w', encoding='utf-8') as f:
    f.write("======================================================\n")
    f.write("       RELATÓRIO DE AVALIAÇÃO DO MODELO HÍBRIDO       \n")
    f.write("======================================================\n\n")
    
    f.write("1. MÉTRICAS DE DESEMPENHO (FILTRO DIURNO > 10 W/m²)\n")
    f.write("-" * 50 + "\n")
    for log in metricas_log:
        f.write(log + "\n")
    
    f.write("\n2. BASE DE REGRAS FUZZY (MAMDANI)\n")
    f.write("-" * 50 + "\n")
    # materializa as regras em uma lista para suportar len() e múltiplas iterações
    regras = list(controle_ghi.rules)
    f.write(f"Total de Regras: {len(regras)}\n\n")
    
    for i, regra in enumerate(regras):
        f.write(f"Regra {i+1}: {regra}\n")

print(f"Relatório salvo em: {txt_filename}")

# Salvar CSV
df_eval.to_csv(f'{OUTPUT_DIR}/resultados_comparativos.csv')

print(f"\nProcesso concluído! Todos os arquivos estão em: {OUTPUT_DIR}")