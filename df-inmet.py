import pandas as pd
import os

# --- 1. CONFIGURAÇÃO ---

# Pasta onde estão os arquivos CSV brutos do INMET
PASTA_DOS_DADOS = '/home/murilo/Área de trabalho/GitHub/solar-ia/data/inmet'

# Lista dos arquivos das estações que decidimos usar (nossas melhores candidatas)
ARQUIVOS_ESTACOES = [
    'dados_A304_H_2018-01-01_2025-06-30.csv', # 30% nulos
    'dados_A316_H_2018-01-01_2025-06-30.csv', # 36% nulos
    'dados_A372_H_2018-01-01_2025-06-30.csv', # 38% nulos
    'dados_A340_H_2018-01-01_2025-06-30.csv', # 40% nulos
]

# Dicionário com as coordenadas de cada estação.
COORDENADAS_ESTACOES = {
    'A304': {'latitude': -5.83722221, 'longitude': -35.20805555}, # Natal
    'A316': {'latitude': -6.4674999, 'longitude': -37.08499999},  # Caicó
    'A372': {'latitude': -5.5349999, 'longitude': -36.87222221},  # Macau
    'A340': {'latitude': -5.6266666, 'longitude': -37.815},       # Apodi
}

NOMES_COLUNAS_INMET = [
    'data', 'hora', 'precipitacao', 'pressao_atm_estacao',
    'pressao_atm_max', 'pressao_atm_min', 'radiacao_global',
    'temp_ar', 'temp_max', 'temp_min',
    'umidade_max', 'umidade_min', 'umidade_rel',
    'vento_dir', 'vento_rajada', 'vento_vel',
    'descartar'
]

COLUNAS_FINAIS_INMET = [
    'codigo_estacao',
    'latitude',
    'longitude',
    'temp_ar',
    'umidade_rel',
    'pressao_atm_estacao',
    'vento_vel',
    'vento_dir',
    'precipitacao'
]

# --- 2. PROCESSAMENTO E UNIFICAÇÃO ---

lista_de_dataframes = []
print("Iniciando a limpeza e unificação dos dados...\n")

for nome_arquivo in ARQUIVOS_ESTACOES:
    codigo_estacao = nome_arquivo.split('_')[1]
    caminho_completo = os.path.join(PASTA_DOS_DADOS, nome_arquivo)
    
    print(f"Processando estação: {codigo_estacao}...")

    df_estacao = pd.read_csv(
        caminho_completo,
        sep=';',
        skiprows=11,
        header=None,
        names=NOMES_COLUNAS_INMET, # Atribui nossa lista de nomes limpos
        decimal='.',
        encoding='latin-1',
        na_values=['null']
    )

    # O dicionário de mapeamento não é mais necessário! O código fica mais limpo.
    
    # Cria o timestamp e o define como índice
    df_estacao['hora'] = df_estacao['hora'].astype(str).str.zfill(4).str.slice(0, 2) + ':00:00'
    df_estacao['timestamp'] = pd.to_datetime(df_estacao['data'] + ' ' + df_estacao['hora'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df_estacao.set_index('timestamp', inplace=True)
    
    # Adiciona as informações de metadados
    df_estacao['codigo_estacao'] = codigo_estacao
    df_estacao['latitude'] = COORDENADAS_ESTACOES[codigo_estacao]['latitude']
    df_estacao['longitude'] = COORDENADAS_ESTACOES[codigo_estacao]['longitude']
    
    # Seleciona apenas as colunas finais
    # O reindex garante que a ordem e a presença das colunas estejam corretas
    df_processado = df_estacao.reindex(columns=COLUNAS_FINAIS_INMET)
    
    lista_de_dataframes.append(df_processado)

df_master_inmet = pd.concat(lista_de_dataframes)
df_master_inmet.sort_index(inplace=True)

print("\nProcesso de unificação concluído!")
print("Amostra do DataFrame Mestre (INMET):")
print(df_master_inmet.head())
print("\nInformações do DataFrame Mestre (INMET):")
df_master_inmet.info()
df_master_inmet.to_parquet('df_master_inmet.parquet')