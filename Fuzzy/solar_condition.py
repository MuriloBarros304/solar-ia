import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ======================================================
# 1. Declaração das variáveis fuzzy
# ======================================================
hora_sin = ctrl.Antecedent(np.linspace(-1, 1, 100), 'hora_sin')
hora_cos = ctrl.Antecedent(np.linspace(-1, 1, 100), 'hora_cos')
tipo_nuvem = ctrl.Antecedent(np.linspace(0, 10, 100), 'tipo_nuvem')
temp_ar = ctrl.Antecedent(np.linspace(0, 40, 100), 'temp_ar')
condicao_solar = ctrl.Consequent(np.linspace(0, 100, 200), 'condicao_solar')

# ======================================================
# 2. Funções de pertinência
# ======================================================

# hora_sin
hora_sin['baixa'] = fuzz.trimf(hora_sin.universe, [-1, -1, 0])
hora_sin['media'] = fuzz.trimf(hora_sin.universe, [-1, 0, 1])
hora_sin['alta'] = fuzz.trimf(hora_sin.universe, [0, 1, 1])

# hora_cos
hora_cos['baixa'] = fuzz.trimf(hora_cos.universe, [-1, -1, 0])
hora_cos['media'] = fuzz.trimf(hora_cos.universe, [-1, 0, 1])
hora_cos['alta'] = fuzz.trimf(hora_cos.universe, [0, 1, 1])

# tipo_nuvem
tipo_nuvem['limpo'] = fuzz.trimf(tipo_nuvem.universe, [0, 0, 3])
tipo_nuvem['parcial'] = fuzz.trimf(tipo_nuvem.universe, [2, 5, 8])
tipo_nuvem['carregado'] = fuzz.trimf(tipo_nuvem.universe, [7, 10, 10])

# temp_ar
temp_ar['baixa'] = fuzz.trimf(temp_ar.universe, [0, 0, 18])
temp_ar['media'] = fuzz.trimf(temp_ar.universe, [15, 25, 32])
temp_ar['alta'] = fuzz.trimf(temp_ar.universe, [28, 40, 40])

# condicao_solar
condicao_solar['baixa'] = fuzz.trimf(condicao_solar.universe, [0, 0, 40])
condicao_solar['media'] = fuzz.trimf(condicao_solar.universe, [30, 50, 70])
condicao_solar['alta'] = fuzz.trimf(condicao_solar.universe, [60, 100, 100])

# ======================================================
# 3. Regras do sistema fuzzy
# ======================================================
regra1 = ctrl.Rule(
    tipo_nuvem['limpo'] & temp_ar['alta'],
    condicao_solar['alta']
)

regra2 = ctrl.Rule(
    tipo_nuvem['parcial'] & temp_ar['media'],
    condicao_solar['media']
)

regra3 = ctrl.Rule(
    tipo_nuvem['carregado'],
    condicao_solar['baixa']
)

regra4 = ctrl.Rule(
    hora_sin['alta'] & hora_cos['alta'] & tipo_nuvem['limpo'],
    condicao_solar['alta']
)

regra5 = ctrl.Rule(
    hora_sin['baixa'] | hora_cos['baixa'],
    condicao_solar['baixa']
)

# Sistema de controle
controle_solar = ctrl.ControlSystem([regra1, regra2, regra3, regra4, regra5])
simulador = ctrl.ControlSystemSimulation(controle_solar)

# ======================================================
# 4. Função de avaliação
# ======================================================
def avaliar_condicao_solar(h_sin, h_cos, nuvem, temp):
    simulador.input['hora_sin'] = h_sin
    simulador.input['hora_cos'] = h_cos
    simulador.input['tipo_nuvem'] = nuvem
    simulador.input['temp_ar'] = temp

    simulador.compute()

    return simulador.output['condicao_solar']


# ======================================================
# 5. Função para gerar gráfico de pertinência de uma variável
# ======================================================
def plot_var(var):
    fig, ax = plt.subplots(figsize=(7, 3))
    for label, mf in var.terms.items():
        ax.plot(var.universe, mf.mf, label=label)
    ax.set_title(f"Funções de Pertinência - {var.label}")
    ax.set_xlabel(var.label)
    ax.set_ylabel("Grau de Pertinência")
    ax.legend()
    ax.grid(True)
    return fig


# ======================================================
# 6. Função para mostrar gráfico da condição solar
# ======================================================
def mostrar_grafico_condicao_solar():
    return plot_var(condicao_solar)


# ======================================================
# 7. Texto interpretando o resultado crisp
# ======================================================
def interpretar_condicao_fuzzy(valor):
    if valor < 40:
        return "🌥 Condição Solar BAIXA"
    elif valor < 70:
        return "⛅ Condição Solar MÉDIA"
    else:
        return "☀ Condição Solar ALTA"
