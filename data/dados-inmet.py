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
        
        # 2. ANÁLISE 1: LACUNAS TEMPORAIS (LINHAS FALTANTES)
        print("\n[Análise 1: Lacunas Temporais (Horas Inteiras Faltando)]")
        data_inicio = df.index.min()
        data_fim = df.index.max()

        if pd.isna(data_inicio):
            print("AVISO: Não foi possível ler datas neste arquivo. Pulando.")
            print("-" * 50 + "\n")
            return

        indice_esperado = pd.date_range(start=data_inicio, end=data_fim, freq='H')
        registros_esperados = len(indice_esperado)
        registros_reais = len(df)
        registros_faltantes = registros_esperados - registros_reais
        percentual_lacunas = (registros_faltantes / registros_esperados) * 100 if registros_esperados > 0 else 0

        print(f"Período de dados: de {data_inicio.strftime('%Y-%m-%d')} a {data_fim.strftime('%Y-%m-%d')}")
        print(f"Total de horas no período: {registros_esperados}")
        print(f"Registros encontrados: {registros_reais}")
        print(f"Percentual de LACUNAS (linhas faltantes): {percentual_lacunas:.2f}%")

        # 3. ANÁLISE 2: VALORES NULOS INTERNOS (CÉLULAS VAZIAS)
        print("\n[Análise 2: Valores Nulos Internos (Dados Faltando Dentro das Linhas)]")
        
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