import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb # Necessário para carregar o modelo XGBoost
import sklearn.metrics

# --- 1. CONFIGURAÇÃO ---
print("Iniciando o módulo de avaliação final dos modelos...")

# Caminhos para os modelos e dados de TESTE
RF_MODEL_PATH = 'training/random_forest_model.joblib'
XGB_GHI_MODEL_PATH = 'training/xgb_model_ghi.joblib'
XGB_DNI_MODEL_PATH = 'training/xgb_model_dni.joblib'

X_TEST_PATH = 'data/X_test.parquet'
Y_TEST_PATH = 'data/y_test.parquet'

# Período para visualização nos gráficos de série temporal (use o ano de 2024)
PLOT_START_DATE = '2024-07-15'
PLOT_END_DATE = '2024-07-18'

# --- 2. FUNÇÕES AUXILIARES ---

def calculate_and_save_metrics(y_true, y_pred, model_name):
    """Calcula e imprime as métricas de erro para GHI e DNI."""
    mae_ghi = sklearn.metrics.mean_absolute_error(y_true['ghi'], y_pred['ghi'])
    rmse_ghi = np.sqrt(sklearn.metrics.mean_squared_error(y_true['ghi'], y_pred['ghi']))
    mae_dni = sklearn.metrics.mean_absolute_error(y_true['dni'], y_pred['dni'])
    rmse_dni = np.sqrt(sklearn.metrics.mean_squared_error(y_true['dni'], y_pred['dni']))

    ghi_medio_dia = y_true[y_true['ghi'] > 0]['ghi'].mean()
    dni_medio_dia = y_true[y_true['dni'] > 0]['dni'].mean()

    with open(f'predict/{model_name.lower()}_performance.txt', 'w') as f:
        f.write(f"Desempenho do Modelo {model_name}:\n")
        f.write("="*50 + "\n")
        f.write(f"Target: GHI\n")
        f.write(f"  - Erro Médio Absoluto (MAE): {mae_ghi:.2f} W/m²\n")
        f.write(f"  - Raiz do Erro Quadrático Médio (RMSE): {rmse_ghi:.2f} W/m²\n")
        f.write("-"*50 + "\n")
        f.write(f"Target: DNI\n")
        f.write(f"  - Erro Médio Absoluto (MAE): {mae_dni:.2f} W/m²\n")
        f.write(f"  - Raiz do Erro Quadrático Médio (RMSE): {rmse_dni:.2f} W/m²\n")
        f.write("="*50 + "\n")
        f.write("\nPara Contexto:\n")
        f.write(f"  - GHI médio (durante o dia) no set de teste: {ghi_medio_dia:.2f} W/m²\n")
        f.write(f"  - DNI médio (durante o dia) no set de teste: {dni_medio_dia:.2f} W/m²\n")

# (Substitua sua função original por esta)

def plot_feature_importance(model, feature_names, model_name):
    """Plota e SALVA a importância das features para modelos baseados em árvores."""
    if not hasattr(model, 'feature_importances_'):
        print(f"\nO modelo {model_name} não suporta 'feature_importances_'.")
        return
        
    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:] # Top 20 features
    
    # Cria uma nova figura para este gráfico específico
    plt.figure(figsize=(10, 12))
    plt.title(f'Importância das Features - {model_name}')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importância Relativa')
    plt.tight_layout()
    
    # Define um nome de arquivo e salva a figura
    filename = f'feature_importance_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(f'predict/{filename}', dpi=300, bbox_inches='tight')
    print(f"- Gráfico de importância das features salvo como '{filename}'")
    plt.close() # Fecha a figura para liberar memória

# --- 3. SCRIPT PRINCIPAL ---

def main():
    # Carregar dados
    print(f"\nCarregando dados de teste de '{X_TEST_PATH}' e '{Y_TEST_PATH}'...")
    try:
        X_test = pd.read_parquet(X_TEST_PATH)
        y_test = pd.read_parquet(Y_TEST_PATH)
    except FileNotFoundError:
        print("ERRO: Arquivos de teste não encontrados. Certifique-se de que o script de preparação e separação foi executado.")
        return

    # Carregar modelos
    print("Carregando modelos treinados...")
    try:
        rf_model = joblib.load(RF_MODEL_PATH)
        xgb_model_ghi = joblib.load(XGB_GHI_MODEL_PATH)
        xgb_model_dni = joblib.load(XGB_DNI_MODEL_PATH)
    except FileNotFoundError:
        print("ERRO: Arquivos de modelo não encontrados. Certifique-se de que os modelos foram treinados e salvos.")
        return
        
    # Gerar previsões
    print("Gerando previsões no conjunto de teste...")
    pred_rf_raw = rf_model.predict(X_test)
    pred_rf = pd.DataFrame(pred_rf_raw, index=y_test.index, columns=y_test.columns)

    pred_xgb_ghi = xgb_model_ghi.predict(X_test)
    pred_xgb_dni = xgb_model_dni.predict(X_test)
    pred_xgb = pd.DataFrame({'ghi': pred_xgb_ghi, 'dni': pred_xgb_dni}, index=y_test.index)

    # Calcular e salvar métricas
    calculate_and_save_metrics(y_test, pred_rf, "RandomForest")
    calculate_and_save_metrics(y_test, pred_xgb, "XGBoost")
    
    # --- PLOTS DE VISUALIZAÇÃO ---
    print("\nGerando gráficos de avaliação...")
    plt.style.use('seaborn-v0_8-whitegrid')

    # Filtrar período para plotagem
    y_test_period = y_test.loc[PLOT_START_DATE:PLOT_END_DATE]
    pred_rf_period = pred_rf.loc[PLOT_START_DATE:PLOT_END_DATE]
    pred_xgb_period = pred_xgb.loc[PLOT_START_DATE:PLOT_END_DATE]

    # Gráfico de Série Temporal
    fig_ts, axs_ts = plt.subplots(nrows=2, ncols=1, figsize=(15, 10), sharex=True)
    fig_ts.suptitle(f'Comparação de Previsões no Set de Teste ({PLOT_START_DATE} a {PLOT_END_DATE})', fontsize=16)
    # GHI
    axs_ts[0].plot(y_test_period.index, y_test_period['ghi'], label='Real', color='black')
    axs_ts[0].plot(pred_rf_period.index, pred_rf_period['ghi'], label='RandomForest', linestyle='--')
    axs_ts[0].plot(pred_xgb_period.index, pred_xgb_period['ghi'], label='XGBoost', linestyle=':')
    axs_ts[0].set_ylabel('GHI (W/m²)')
    axs_ts[0].legend()
    # DNI
    axs_ts[1].plot(y_test_period.index, y_test_period['dni'], label='Real', color='black')
    axs_ts[1].plot(pred_rf_period.index, pred_rf_period['dni'], label='RandomForest', linestyle='--')
    axs_ts[1].plot(pred_xgb_period.index, pred_xgb_period['dni'], label='XGBoost', linestyle=':')
    axs_ts[1].set_xlabel('Data e Hora')
    axs_ts[1].set_ylabel('DNI (W/m²)')
    axs_ts[1].legend()
    fig_ts.autofmt_xdate()
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))

    # SALVA O GRÁFICO DE SÉRIE TEMPORAL
    plt.savefig('predict/comparacao_series_temporais.png', dpi=300, bbox_inches='tight')
    print("- Gráfico de séries temporais salvo como 'comparacao_series_temporais.png'")
    plt.close(fig_ts) # Fecha a figura para liberar memória
    
    # Gráfico de Dispersão (Real vs. Previsto)
    fig_scatter, axs_scatter = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
    fig_scatter.suptitle('Análise de Erro: Real vs. Previsto', fontsize=16)
    max_val = max(y_test['ghi'].max(), y_test['dni'].max())
    # GHI
    axs_scatter[0].scatter(y_test['ghi'], pred_rf['ghi'], label='RandomForest', alpha=0.3)
    axs_scatter[0].scatter(y_test['ghi'], pred_xgb['ghi'], label='XGBoost', alpha=0.3)
    axs_scatter[0].plot([0, max_val], [0, max_val], color='black', linestyle='--', label='Linha Perfeita (y=x)')
    axs_scatter[0].set_xlabel('GHI Real (W/m²)')
    axs_scatter[0].set_ylabel('GHI Previsto (W/m²)')
    axs_scatter[0].set_title('GHI')
    axs_scatter[0].legend()
    axs_scatter[0].grid(True)
    # DNI
    axs_scatter[1].scatter(y_test['dni'], pred_rf['dni'], label='RandomForest', alpha=0.3)
    axs_scatter[1].scatter(y_test['dni'], pred_xgb['dni'], label='XGBoost', alpha=0.3)
    axs_scatter[1].plot([0, max_val], [0, max_val], color='black', linestyle='--', label='Linha Perfeita (y=x)')
    axs_scatter[1].set_xlabel('DNI Real (W/m²)')
    axs_scatter[1].set_ylabel('DNI Previsto (W/m²)')
    axs_scatter[1].set_title('DNI')
    axs_scatter[1].legend()
    axs_scatter[1].grid(True)

    # SALVA O GRÁFICO DE DISPERSÃO
    plt.savefig('predict/analise_dispersao_erro.png', dpi=300, bbox_inches='tight')
    print("- Gráfico de dispersão salvo como 'analise_dispersao_erro.png'")
    plt.close(fig_scatter) # Fecha a figura para liberar memória
    
    # Gráficos de Importância das Features
    plot_feature_importance(rf_model, X_test.columns, 'RandomForest')
    
    # Para o XGBoost, podemos pegar a importância do modelo GHI como representativa
    plot_feature_importance(xgb_model_ghi, X_test.columns, 'XGBoost (GHI model)')
    
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    # plt.show()

if __name__ == '__main__':
    main()