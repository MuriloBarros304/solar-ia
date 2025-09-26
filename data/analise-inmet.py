import pandas as pd
import os

# --- CONFIGURAÇÃO ---
PASTA_DOS_DADOS = '/home/murilo/Área de trabalho/GitHub/solar-ia/data/inmet'

ARQUIVOS_PARA_ANALISAR = ['dados_A304_H_2018-01-01_2025-06-30.csv', 'dados_A316_H_2018-01-01_2025-06-30.csv',
                         'dados_A372_H_2018-01-01_2025-06-30.csv', 'dados_A340_H_2018-01-01_2025-06-30.csv']

# Número de linhas do cabeçalho que devem ser puladas
LINHAS_CABECALHO = 10

def analise_detalhada_colunas(caminho_arquivo):
    """
    Realiza uma análise detalhada, mostrando o percentual de dados nulos
    para cada coluna de um arquivo de estação do INMET.
    """
    try:
        nome_estacao = os.path.basename(caminho_arquivo).split('.')[0]
        print(f"--- Estação: {nome_estacao} ---")

        # Carrega o CSV, tratando a string 'null' como um valor nulo (NaN)
        df = pd.read_csv(
            caminho_arquivo,
            sep=';',
            skiprows=LINHAS_CABECALHO,
            decimal='.',
            encoding='latin-1',
            na_values=['null']
        )
        
        # Se o DataFrame estiver vazio, não há o que fazer.
        if df.empty:
            print("Arquivo vazio ou não foi possível ler os dados.")
            return

        total_linhas = len(df)
        print(f"Total de linhas (horas) no arquivo: {total_linhas}\n")
        print("Relatório de valores nulos por coluna (sensor):")
        print("-" * 50)

        # Calcula o percentual de nulos para cada coluna
        nulos_por_coluna = df.isnull().sum()
        percentual_nulos = (nulos_por_coluna / total_linhas) * 100

        # Ordena o resultado para mostrar as colunas mais problemáticas primeiro
        percentual_nulos_ordenado = percentual_nulos.sort_values(ascending=False)

        # Imprime o relatório formatado
        for coluna, percentual in percentual_nulos_ordenado.items():
            # A formatação abaixo ajuda a alinhar o texto para fácil leitura
            print(f"{coluna:<50} | {percentual:>7.2f}% nulos")
            
        print("-" * 50)

    except FileNotFoundError:
        print(f"ERRO: O arquivo '{caminho_arquivo}' não foi encontrado.")
        print("Verifique se o nome do arquivo e o caminho da pasta estão corretos na configuração.")
    except Exception as e:
        print(f"ERRO inesperado ao processar o arquivo {caminho_arquivo}: {e}")


if __name__ == '__main__':
    for arquivo in ARQUIVOS_PARA_ANALISAR:
        caminho_completo = os.path.join(PASTA_DOS_DADOS, arquivo)
        analise_detalhada_colunas(caminho_completo)