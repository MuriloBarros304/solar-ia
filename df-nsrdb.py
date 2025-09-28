import pandas as pd
import os

# --- 1. CONFIGURAÇÃO ---

# Pasta raiz onde as subpastas das estações (a304, a316...) estão localizadas
PASTA_DADOS_NSRDB = '/home/murilo/Área de trabalho/GitHub/solar-ia/data/nsrdb'

# Dicionário de coordenadas. Os códigos (chaves) devem corresponder aos nomes das subpastas em minúsculo.
COORDENADAS_ESTACOES = {
    'a304': {'latitude': -5.837222, 'longitude': -35.208056}, 
    'a316': {'latitude': -6.467500, 'longitude': -37.085000},
    'a372': {'latitude': -5.535000, 'longitude': -36.872222}, 
    'a340': {'latitude': -5.626677, 'longitude': -37.815000}  
}

# Mapeamento atualizado com as novas colunas que você adicionou
MAPEAMENTO_COLUNAS_NSRDB = {
    'GHI': 'ghi',
    'DNI': 'dni',
    'DHI': 'dhi',
    'Temperature': 'temp_ar_nsrdb',
    'Relative Humidity': 'umidade_rel_nsrdb',
    'Wind Speed': 'vento_vel_nsrdb',
    'Cloud Type': 'tipo_nuvem_nsrdb', # Nomeado para clareza
    'Pressure': 'pressao_nsrdb'
}

# Lista final de colunas que queremos no nosso DataFrame, incluindo as novas
COLUNAS_FINAIS_NSRDB = [
    'latitude',
    'longitude',
    'ghi',
    'dni',
    'dhi',
    'temp_ar_nsrdb',
    'umidade_rel_nsrdb',
    'vento_vel_nsrdb',
    'tipo_nuvem_nsrdb',
    'pressao_nsrdb'
]

# --- 2. PROCESSAMENTO E UNIFICAÇÃO (LÓGICA ADAPTADA) ---

lista_de_dataframes_nsrdb = []
print("Iniciando a limpeza e unificação dos dados da NSRDB (lendo subpastas)...\n")

# Loop externo: itera por cada estação definida no dicionário de coordenadas
for codigo_estacao, coords in COORDENADAS_ESTACOES.items():
    
    # Monta o caminho para a subpasta da estação (ex: .../nsrdb/a304)
    caminho_pasta_estacao = os.path.join(PASTA_DADOS_NSRDB, codigo_estacao)
    
    print(f"--- Processando Estação: {codigo_estacao.upper()} ---")

    # Verifica se a pasta da estação realmente existe
    if not os.path.isdir(caminho_pasta_estacao):
        print(f"AVISO: A pasta '{caminho_pasta_estacao}' não foi encontrada. Pulando estação.")
        continue

    # Lista para guardar os dataframes de cada ano DENTRO de uma estação
    lista_dfs_anuais = []
    
    # Loop interno: itera por cada arquivo CSV (cada ano) dentro da subpasta da estação
    for nome_arquivo_ano in sorted(os.listdir(caminho_pasta_estacao)):
        if nome_arquivo_ano.endswith('.csv'):
            caminho_completo_ano = os.path.join(caminho_pasta_estacao, nome_arquivo_ano)
            print(f"Lendo arquivo: {nome_arquivo_ano}...")
            
            # Carrega o CSV do ano específico
            try:
                df_ano = pd.read_csv(caminho_completo_ano, skiprows=2)
                lista_dfs_anuais.append(df_ano)
            except Exception as e:
                print(f"ERRO ao ler o arquivo {nome_arquivo_ano}: {e}")

    # Se a lista de dataframes anuais não estiver vazia, continue o processamento
    if lista_dfs_anuais:
        # Concatena todos os arquivos anuais em um único dataframe para a estação
        df_nsrdb = pd.concat(lista_dfs_anuais, ignore_index=True)

        # --- O restante do processo é o mesmo de antes ---
        
        # 1. Criação do Timestamp e tratamento de fuso horário
        df_nsrdb['timestamp'] = pd.to_datetime(df_nsrdb[['Year', 'Month', 'Day', 'Hour', 'Minute']])
        df_nsrdb.set_index('timestamp', inplace=True)
        
        # 2. Renomeia e seleciona as colunas
        df_nsrdb.rename(columns=MAPEAMENTO_COLUNAS_NSRDB, inplace=True)
        
        # 3. Adiciona as coordenadas
        df_nsrdb['latitude'] = coords['latitude']
        df_nsrdb['longitude'] = coords['longitude']
        
        # 4. Filtra para manter apenas as colunas finais
        df_processado = df_nsrdb.reindex(columns=COLUNAS_FINAIS_NSRDB)
        
        # Adiciona o dataframe completo da estação à lista principal
        lista_de_dataframes_nsrdb.append(df_processado)

# 5. Concatena os dataframes de TODAS as estações em um único DataFrame Mestre
df_master_nsrdb = pd.concat(lista_de_dataframes_nsrdb)
df_master_nsrdb.sort_index(inplace=True)

print("\nProcesso de unificação da NSRDB concluído!")
print("Amostra do DataFrame Mestre (NSRDB):")
print(df_master_nsrdb.head())
print("\nInformações do DataFrame Mestre (NSRDB):")
df_master_nsrdb.info()
df_master_nsrdb.to_parquet('data/df_nsrdb.parquet')