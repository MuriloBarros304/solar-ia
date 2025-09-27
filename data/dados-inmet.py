import pandas as pd
import numpy as np
import os

PASTA_DOS_DADOS = '/home/murilo/Área de trabalho/GitHub/solar-ia/data/inmet'
LINHAS_CABECALHO = 10
COLUNA_DATA = 'Data Medicao'
COLUNA_HORA = 'Hora Medicao'


def analisar_estacao(caminho_arquivo):
    """
    Carrega os dados de uma estação, analisa DOIS tipos de dados faltantes
    (linhas ausentes e valores nulos internos) e imprime um resumo completo.
    """
    try:
        nome_estacao = os.path.basename(caminho_arquivo).split('.')[0]
        print(f"--- Analisando Estação: {nome_estacao} ---")

        df = pd.read_csv(
            caminho_arquivo,
            sep=';',
            skiprows=LINHAS_CABECALHO,
            decimal='.',
            encoding='latin-1',
            na_values=['null']
        )

        # 1. Limpeza e Preparação dos Dados
        # Converte a hora de 'HHMM' para 'HH:MM:SS'
        # Adicionado .str antes de zfill para garantir que a coluna seja tratada como string
        df[COLUNA_HORA] = df[COLUNA_HORA].astype(str).str.zfill(4).str.slice(0, 2) + ':00:00'
        
        # Cria a coluna 'timestamp' combinando data e hora com formato corrigido
        # A data do INMET costuma ser AAAA-MM-DD, então este formato é mais robusto
        df['timestamp'] = pd.to_datetime(df[COLUNA_DATA] + ' ' + df[COLUNA_HORA], format='%Y-%m-%d %H:%M:%S', errors='coerce')

        # Remove linhas onde o timestamp não pôde ser criado (dados corrompidos de data/hora)
        df.dropna(subset=['timestamp'], inplace=True)
        df.set_index('timestamp', inplace=True)

        # 2. VALORES NULOS INTERNOS (CÉLULAS VAZIAS)
        print("\n[Valores Nulos Internos (Dados Faltando Dentro das Linhas)]")
        
        # Seleciona apenas as colunas de medição (excluindo as de data/hora originais)
        colunas_medicao = df.columns.drop([COLUNA_DATA, COLUNA_HORA], errors='ignore')
        
        # Calcula o total de células de dados
        total_celulas = df[colunas_medicao].size
        
        # Calcula o total de células nulas
        celulas_nulas = df[colunas_medicao].isnull().sum().sum()
        
        percentual_nulos_internos = (celulas_nulas / total_celulas) * 100 if total_celulas > 0 else 0
        
        print(f"Total de células de medição analisadas: {total_celulas}")
        print(f"Células com valores nulos ('null' ou vazio): {celulas_nulas}")
        print(f"Percentual de VALORES NULOS INTERNOS: {percentual_nulos_internos:.2f}%")
        print("-" * 50 + "\n")

    except Exception as e:
        print(f"ERRO ao processar o arquivo {caminho_arquivo}: {e}")
        print("-" * 50 + "\n")


if __name__ == '__main__':
    print("Iniciando análise qualitativa dos dados das estações INMET...\n")
    if not os.path.isdir(PASTA_DOS_DADOS):
        print(f"ERRO: A pasta '{PASTA_DOS_DADOS}' não foi encontrada.")
    else:
        for nome_arquivo in sorted(os.listdir(PASTA_DOS_DADOS)):
            if nome_arquivo.lower().endswith('.csv'):
                caminho_completo = os.path.join(PASTA_DOS_DADOS, nome_arquivo)
                analisar_estacao(caminho_completo)