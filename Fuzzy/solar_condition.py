import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import streamlit as st

def avaliar_condicao_solar(ghi_value):
    """
    Avalia qualitativamente a condição solar (baixa, média, alta)
    com base no valor de GHI previsto.
    """
    # Variável fuzzy de entrada (GHI)
    ghi = ctrl.Antecedent(np.arange(0, 1100, 1), 'ghi')
    
    # Variável fuzzy de saída (condição solar)
    condicao = ctrl.Consequent(np.arange(0, 100, 1), 'condicao')
    
    # Funções de pertinência
    ghi['baixa'] = fuzz.trimf(ghi.universe, [0, 0, 400])
    ghi['media'] = fuzz.trimf(ghi.universe, [300, 600, 800])
    ghi['alta'] = fuzz.trimf(ghi.universe, [700, 1000, 1000])
    
    condicao['ruim'] = fuzz.trimf(condicao.universe, [0, 0, 40])
    condicao['boa'] = fuzz.trimf(condicao.universe, [30, 60, 80])
    condicao['excelente'] = fuzz.trimf(condicao.universe, [70, 100, 100])
    
    # Regras fuzzy
    regras = [
        ctrl.Rule(ghi['baixa'], condicao['ruim']),
        ctrl.Rule(ghi['media'], condicao['boa']),
        ctrl.Rule(ghi['alta'], condicao['excelente'])
    ]
    
    # Cria o sistema fuzzy
    sistema_ctrl = ctrl.ControlSystem(regras)
    sistema = ctrl.ControlSystemSimulation(sistema_ctrl)
    
    # Entrada
    sistema.input['ghi'] = ghi_value
    sistema.compute()
    
    return sistema.output['condicao']


def interpretar_condicao_fuzzy(valor):
    """
    Traduz o valor fuzzy (0–100) em uma descrição mais realista e gradual.
    """
    if valor < 25:
        return "☁️ Muito ruim (radiação muito baixa)"
    elif 25 <= valor < 40:
        return "🌥️ Não muito boa (radiação baixa)"
    elif 40 <= valor < 55:
        return "⛅ Regular (radiação moderada)"
    elif 55 <= valor < 70:
        return "🌤️ Boa (radiação média)"
    elif 70 <= valor < 85:
        return "☀️ Muito boa (radiação alta)"
    else:
        return "🌞 Excelente (radiação muito alta)"


def mostrar_grafico_condicao_solar(ghi_value):
    """
    Exibe no Streamlit as funções de pertinência da lógica fuzzy
    e o ponto correspondente ao valor de GHI previsto.
    """
    x_ghi = np.arange(0, 1100, 1)
    rad_baixa = fuzz.trimf(x_ghi, [0, 0, 400])
    rad_media = fuzz.trimf(x_ghi, [300, 600, 800])
    rad_alta = fuzz.trimf(x_ghi, [700, 1000, 1000])

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(x_ghi, rad_baixa, 'b', label='Baixa')
    ax.plot(x_ghi, rad_media, 'g', label='Média')
    ax.plot(x_ghi, rad_alta, 'r', label='Alta')
    ax.axvline(ghi_value, color='k', linestyle='--', linewidth=1.5, label=f"GHI={ghi_value:.1f}")
    ax.set_title("Funções de Pertinência - GHI")
    ax.set_xlabel("GHI (W/m²)")
    ax.set_ylabel("Pertinência")
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)
