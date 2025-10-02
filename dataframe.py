import pandas as pd
import numpy as np

HORA_INICIO_DIA = 7
HORA_FIM_DIA = 17
LIMITE_GHI_ANOMALO = 10

try:
    df_inmet = pd.read_parquet('data/df_inmet.parquet')
    df_nsrdb = pd.read_parquet('data/df_nsrdb.parquet')
    print("DataFrames carregados com sucesso.")
except FileNotFoundError as e:
    print(f"ERRO: Arquivo não encontrado. Certifique-se de executar os scripts 'df-inmet.py' e 'df-nsrdb.py' primeiro.")
    print(e)
    exit()

# Tratamento de anomalias nos dados de irradiação solar --------------
is_daylight = (df_nsrdb.index.hour >= HORA_INICIO_DIA) & (df_nsrdb.index.hour <= HORA_FIM_DIA) # type: ignore
is_low_ghi = (df_nsrdb['ghi'] < LIMITE_GHI_ANOMALO)
anomalies_idx = df_nsrdb[is_daylight & is_low_ghi].index
print(f"Encontrados {len(anomalies_idx.unique())} timestamps únicos com anomalias para corrigir.")

# 2. Marcar os valores de irradiação anômalos como NaN
colunas_para_corrigir = ['ghi', 'dni', 'dhi']
if not anomalies_idx.empty:
    df_nsrdb.loc[anomalies_idx, colunas_para_corrigir] = np.nan

# 3. Preencher TODOS os novos buracos com interpolação baseada no tempo
df_nsrdb[colunas_para_corrigir] = df_nsrdb[colunas_para_corrigir].interpolate(method='time')

df_final = df_inmet.join(df_nsrdb, lsuffix='_inmet', rsuffix='_nsrdb')

# Lógica compacta para remover duplicados por grupo (timestamp, codigo_estacao) --------------
# 1. Transforma o índice 'timestamp' em coluna.
# 2. Agrupa por 'timestamp' e 'codigo_estacao'.
# 3. Calcula a média de todas as outras colunas numéricas para cada grupo (colapsando os duplicados).
# 4. Reseta o índice para transformar 'timestamp' e 'codigo_estacao' de volta em colunas.
# 5. Define 'timestamp' como o novo índice.
df_final = (df_final.reset_index()
                    .groupby(['timestamp', 'codigo_estacao'])
                    .mean()
                    .reset_index()
                    .set_index('timestamp'))

# Lista de colunas do INMET para preencher e suas correspondentes da NSRDB
colunas_para_imputar = {
    'temp_ar': 'temp_ar_nsrdb',
    'umidade_rel': 'umidade_rel_nsrdb',
    'vento_vel': 'vento_vel_nsrdb',
    'pressao_atm_estacao': 'pressao_nsrdb',
    }

for col_inmet, col_nsrdb in colunas_para_imputar.items():
    # Conta quantos nulos existem ANTES do preenchimento
    nulos_antes = df_final[col_inmet].isnull().sum()
    
    # Preenche os nulos da coluna INMET com os valores da coluna NSRDB
    df_final[col_inmet] = df_final[col_inmet].fillna(df_final[col_nsrdb])
    
    # Conta quantos nulos sobraram DEPOIS
    nulos_depois = df_final[col_inmet].isnull().sum()
    
    print(f"  - Coluna '{col_inmet}': {nulos_antes - nulos_depois} valores preenchidos. ({nulos_depois} nulos restantes)")

# Remove as colunas de suporte da NSRDB que já usamos
colunas_nsrdb_para_remover = [
    'latitude_nsrdb', 'longitude_nsrdb', 'temp_ar_nsrdb', 
    'umidade_rel_nsrdb', 'vento_vel_nsrdb', 'pressao_nsrdb'
]
df_final.drop(columns=colunas_nsrdb_para_remover, inplace=True, errors='ignore')

# Pode haver alguns poucos nulos restantes se a NSRDB também tiver falhas.
# Usar interpolação para preencher qualquer buraco minúsculo que sobrou.
df_final.interpolate(method='time', limit_direction='both', inplace=True)

# Remove qualquer linha que ainda possa ter nulos (muito improvável, mas é uma boa prática)
df_final.dropna(inplace=True)

# Garante que o índice é do tipo DatetimeIndex
if not isinstance(df_final.index, pd.DatetimeIndex):
    df_final.index = pd.to_datetime(df_final.index)

horas_do_dia = df_final.index.hour
dias_do_ano = df_final.index.dayofyear

# Engenharia de features ----------
df_final['hora_sin'] = np.sin(2 * np.pi * horas_do_dia / 24.0)
df_final['hora_cos'] = np.cos(2 * np.pi * horas_do_dia / 24.0)
df_final['dia_ano_sin'] = np.sin(2 * np.pi * dias_do_ano / 365.25)
df_final['dia_ano_cos'] = np.cos(2 * np.pi * dias_do_ano / 365.25)

# Features de Lag (Defasagem)
lags_a_criar = {
    'temp_ar': [1, 24],
    'umidade_rel': [3],
    'pressao_atm_estacao': [4],
    'dhi': [1, 24],
    'dni': [1, 24],
    'ghi': [1, 24]
}

for coluna, lista_lags in lags_a_criar.items():
    for lag in lista_lags:
        nome_nova_coluna = f'{coluna}_lag{lag}h'
        df_final[nome_nova_coluna] = df_final.groupby('codigo_estacao')[coluna].shift(lag)

# Features de Janela Móvel (Rolling)
colunas_rolling = ['temp_ar', 'umidade_rel', 'pressao_atm_estacao', 'dhi', 'dni', 'ghi']
window_size = 3

for coluna in colunas_rolling:
    # Média Móvel
    nome_media = f'{coluna}_media_movel_{window_size}h'
    df_final[nome_media] = df_final.groupby('codigo_estacao')[coluna].transform(
        lambda x: x.shift(1).rolling(window=window_size).mean()
    )

    # Desvio Padrão Móvel
    nome_std = f'{coluna}_std_movel_{window_size}h'
    df_final[nome_std] = df_final.groupby('codigo_estacao')[coluna].transform(
        lambda x: x.shift(1).rolling(window=window_size).std()
    )

# Remove quaisquer linhas que ainda possam ter nulos após a criação das novas features
df_final.dropna(inplace=True)

print("Amostra do DataFrame Final e Completo:")
print(df_final.head())
print("\nInformações do DataFrame Final e Completo:")
df_final.info()

# Salva o dataset final, pronto para ser usado pelos modelos
df_final.to_parquet('data/dataframe.parquet')
print("Salvo com sucesso!")

# Separação do dataframe final em conjuntos de treino, validação e teste -------

print("\n--- Iniciando a Separação dos Dados (Treino, Validação, Teste) ---")

# Remover os dados de 2025 que não têm alvo correspondente
df_final = df_final.loc[df_final.index < pd.to_datetime('2025-01-01')]
print(f"Dataset finalizado. Período total: de {df_final.index.min()} a {df_final.index.max()}")

# Divisão Cronológica
train_df = df_final.loc[df_final.index < '2023-01-01']
val_df = df_final.loc[(df_final.index >= '2023-01-01') & (df_final.index < '2024-01-01')]
test_df = df_final.loc[df_final.index >= '2024-01-01']

print(f"\nRegistros de Treino: {len(train_df)} ({len(train_df) / len(df_final) * 100:.1f}%)")
print(f"Registros de Validação: {len(val_df)} ({len(val_df) / len(df_final) * 100:.1f}%)")
print(f"Registros de Teste: {len(test_df)} ({len(test_df) / len(df_final) * 100:.1f}%)")

# Nossos alvos são 'ghi' e 'dni'. Todas as outras colunas são features.
FEATURES = [col for col in df_final.columns if col not in ['ghi', 'dni', 'codigo_estacao', 'dhi']]
TARGETS = ['ghi', 'dni']

X_train = train_df[FEATURES]
y_train = train_df[TARGETS]

X_val = val_df[FEATURES]
y_val = val_df[TARGETS]

X_test = test_df[FEATURES]
y_test = test_df[TARGETS]

print("\nSeparação em X (features) e y (alvos) concluída.")
print(f"Shape de X_train: {X_train.shape}")
print(f"Shape de y_train: {y_train.shape}")
print(f"Shape de X_val: {X_val.shape}")
print(f"Shape de y_val: {y_val.shape}")
print(f"Shape de X_test: {X_test.shape}")
print(f"Shape de y_test: {y_test.shape}")

X_train.to_parquet('data/X_train.parquet')
y_train.to_parquet('data/y_train.parquet')
X_val.to_parquet('data/X_val.parquet')
y_val.to_parquet('data/y_val.parquet')
X_test.to_parquet('data/X_test.parquet')
y_test.to_parquet('data/y_test.parquet')