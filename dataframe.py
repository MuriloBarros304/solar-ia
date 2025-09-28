import pandas as pd


try:
    df_inmet = pd.read_parquet('df_master_inmet.parquet')
    df_nsrdb = pd.read_parquet('df_master_nsrdb.parquet')
    print("DataFrames carregados com sucesso.")
except FileNotFoundError as e:
    print(f"ERRO: Arquivo não encontrado. Certifique-se de executar os scripts 'df-inmet.py' e 'df-nsrdb.py' primeiro.")
    print(e)
    exit()

df_final = df_inmet.join(df_nsrdb, lsuffix='_inmet', rsuffix='_nsrdb')


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

print("Limpeza finalizada.")

print("\nProcesso concluído! O dataset final está pronto para modelagem.")
print("Amostra do DataFrame Final e Completo:")
print(df_final.head())
print("\nInformações do DataFrame Final e Completo:")
df_final.info()

# Salva o dataset final, pronto para ser usado pelos modelos
print("\nSalvando DataFrame final em 'dataset_completo.parquet'...")
df_final.to_parquet('dataset_completo.parquet')
print("Salvo com sucesso!")