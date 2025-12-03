"""
Módulo para avaliação e comparação dos sistemas fuzzy (Mamdani e Sugeno)
com as predições de Machine Learning (XGBoost).
Métrica unificada: W/m² (Irradiação Global Horizontal)
"""

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

from Fuzzy.ghi_mamdani import (
    avaliar_ghi_mamdani,
    hora_sin, hora_cos, tipo_nuvem, temp_ar, ghi, controle_ghi
)
from Fuzzy.ghi_sugeno import avaliar_ghi_sugeno

# Configuração de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

OUTPUT_DIR = 'predict/fuzzy_evaluation'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("AVALIAÇÃO: ML vs FUZZY (GHI W/m²)")
print("=" * 70)

# ======================================================
# 1. CARREGAMENTO DOS DADOS
# ======================================================
print("\n[1/8] Carregando dados e models...")
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

print(f"Amostra definida: {len(X_sample)} pontos")

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
df_eval['XGBoost'] = y_xgb_sample
df_eval['Mamdani'] = mamdani_preds
df_eval['Sugeno'] = sugeno_preds

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
    
    msg = f"--- {name} ---\nMAE:  {mae:.2f} W/m²\nRMSE: {rmse:.2f} W/m²\nR²:   {r2:.4f}\n"
    print(msg)
    metricas_log.append(msg)
    
    return mae, rmse, r2

mae_ml, rmse_ml, r2_ml = calc_metrics(df_diurno['GHI_Real'], df_diurno['XGBoost'], "XGBoost")
mae_mam, rmse_mam, r2_mam = calc_metrics(df_diurno['GHI_Real'], df_diurno['Mamdani'], "Mamdani")
mae_sug, rmse_sug, r2_sug = calc_metrics(df_diurno['GHI_Real'], df_diurno['Sugeno'], "Sugeno")

# ======================================================
# 6. GERAÇÃO DE GRÁFICOS
# ======================================================
print("\n[6/8] Gerando gráficos comparativos...")

# Série Temporal
plt.figure(figsize=(14, 6))
subset = df_eval.iloc[50:150]
plt.plot(subset.index, subset['GHI_Real'], 'k-', linewidth=2, label='Real', alpha=0.8)
plt.plot(subset.index, subset['XGBoost'], 'b--', label='XGBoost', alpha=0.7)
plt.plot(subset.index, subset['Mamdani'], 'orange', label='Mamdani', alpha=0.8)
plt.plot(subset.index, subset['Sugeno'], 'r:', linewidth=2, label='Sugeno', alpha=0.8)
plt.title('Comparação Temporal das Predições de GHI')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f'{OUTPUT_DIR}/series_temporal_ghi.png', dpi=300)
plt.close()

# Scatter Plots
plt.figure(figsize=(18, 5))
models = ['XGBoost', 'Mamdani', 'Sugeno']
colors = ['b--', 'orange', 'r:']
for i, model in enumerate(models, 1):
    plt.subplot(1, 3, i)
    plt.scatter(df_diurno['GHI_Real'], df_diurno[model], alpha=0.4)
    plt.plot([0, 1000], [0, 1000], colors[i-1], label=model, alpha=0.7)
    plt.title(f'Real vs. {model}')
    plt.xlabel('GHI Real (W/m²)')
    plt.ylabel(f'GHI Predito ({model})')
    plt.xlim(0, 1000)
    plt.ylim(0, 1000)
    plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/scatter_real_vs_predito.png', dpi=300)
plt.close()

# Barras Métricas
plt.figure(figsize=(10, 6))
maes = [mae_ml, mae_mam, mae_sug]
rmses = [rmse_ml, rmse_mam, rmse_sug]
x = np.arange(len(models))
w = 0.35
plt.bar(x - w/2, maes, w, label='MAE', color=['#90CAF9', '#FFCC80', '#EF9A9A'])
plt.bar(x + w/2, rmses, w, label='RMSE', color=['#1976D2', '#F57C00', '#D32F2F'])
plt.xticks(x, models)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.savefig(f'{OUTPUT_DIR}/barras_metricas.png', dpi=300)
plt.close()

# ======================================================
# 7. SALVANDO AS FUNÇÕES DE PERTINÊNCIA (INPUTS E SAÍDA)
# ======================================================
matplotlib.use('Agg') # Garante que não abra janelas

print("\n[7/8] Salvando funções de pertinência (Modo Manual)...")

# Lista das variáveis
variaveis_entrada_fuzzy = [
    (hora_sin,  'Input - Ciclo Diário (Seno)'),
    (hora_cos,  'Input - Dia/Noite (Cosseno)'),
    (tipo_nuvem,'Input - Cobertura de Nuvens'),
    (temp_ar,   'Input - Temperatura do Ar')
]

# Configuração da Grade de Subplots
n = len(variaveis_entrada_fuzzy)
cols = 2
rows = int(np.ceil(n / cols))

fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows))
axes = axes.flatten()

# Cores para as linhas (opcional, para ficar bonito)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

n_vars = len(variaveis_entrada_fuzzy)

for i, (var, titulo) in enumerate(variaveis_entrada_fuzzy):
    ax = axes[i]
    
    # --- AQUI ESTÁ A MUDANÇA MÁGICA ---
    # Em vez de var.view(ax=ax), nós iteramos sobre os termos
    # var.terms contém as etiquetas ('baixa', 'media', etc.)
    # var.universe contém o eixo X
    # var[label].mf contém o eixo Y (membership function)
    
    color_idx = 0
    for label in var.terms:
        # Pega a cor atual da lista ciclicamente
        cor = colors[color_idx % len(colors)]
        
        # Plota a linha manualmente
        ax.plot(var.universe, var[label].mf, label=label, linewidth=2, color=cor)
        
        # Preenche embaixo da curva (opcional, igual ao view padrão)
        ax.fill_between(var.universe, var[label].mf, alpha=0.1, color=cor)
        
        color_idx += 1

    ax.set_title(titulo, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05) # Margem para ver bem o topo e base

# Remove eixos vazios se houver
for j in range(n_vars, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fuzzy_inputs.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print("  -> Salvo: fuzzy_inputs.png")

# --- 7.2 Saída (Também manual para garantir consistência) ---
fig2, ax2 = plt.subplots(figsize=(10, 6))

# Plotando GHI manualmente
color_idx = 0
for label in ghi.terms:
    cor = colors[color_idx % len(colors)]
    ax2.plot(ghi.universe, ghi[label].mf, label=label, linewidth=2, color=cor)
    ax2.fill_between(ghi.universe, ghi[label].mf, alpha=0.1, color=cor)
    color_idx += 1

ax2.set_title("Output - GHI (W/m²)", fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fuzzy_output.png", dpi=300, bbox_inches='tight')
plt.close(fig2)

print("  -> Salvo: fuzzy_output.png")

# ======================================================
# 8. GERAR RELATÓRIO TXT
# ======================================================
print("\n[8/8] Gerando relatório de texto (Métricas e Regras)...")

txt_filename = f'{OUTPUT_DIR}/relatorio_metricas_regras.txt'
with open(txt_filename, 'w', encoding='utf-8') as f:
    f.write("======================================================\n")
    f.write("       RELATÓRIO DE AVALIAÇÃO DO SISTEMA FUZZY        \n")
    f.write("======================================================\n\n")
    
    f.write("1. MÉTRICAS DE DESEMPENHO\n")
    f.write("-" * 50 + "\n")
    for log in metricas_log:
        f.write(log + "\n")
    
    f.write("\n2. BASE DE REGRAS FUZZY\n")
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