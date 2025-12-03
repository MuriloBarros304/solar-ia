import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ======================================================
# 1. DEFINIÇÃO DAS VARIÁVEIS
# ======================================================

# HORA COSSENO (Elevação Solar)
# Universo: -1 (Meio-dia) a 1 (Meia-noite)
hora_cos = ctrl.Antecedent(np.linspace(-1, 1, 500), 'hora_cos')

# HORA SENO (Assimetria Manhã/Tarde)
# Universo: -1 (18h) a 1 (06h)
hora_sin = ctrl.Antecedent(np.linspace(-1, 1, 500), 'hora_sin')

# TIPO DE NUVEM
# Universo: 0 (Limpo) a 10 (Fechado)
tipo_nuvem = ctrl.Antecedent(np.linspace(0, 10, 200), 'tipo_nuvem')

# TEMPERATURA
# Universo: 10°C a 45°C
temp_ar = ctrl.Antecedent(np.linspace(10, 45, 200), 'temp_ar')

# OUTPUT: GHI (W/m²)
ghi = ctrl.Consequent(np.linspace(0, 1000, 1000), 'ghi', defuzzify_method='centroid')


# ======================================================
# 2. FUNÇÕES DE PERTINÊNCIA 
# ======================================================

# --- HORA_COS (Elevação) ---
# Foco na região negativa (dia).
# -1.0 = Zênite (Sol a pino)
# -0.5 = 45 graus (Sol alto)
#  0.0 = Horizonte
hora_cos['zenite']    = fuzz.trimf(hora_cos.universe, [-1.0, -1.0, -0.5])
hora_cos['alto']      = fuzz.trimf(hora_cos.universe, [-0.8, -0.4, 0.0])
hora_cos['baixo']     = fuzz.trimf(hora_cos.universe, [-0.3, 0.0, 0.3])
hora_cos['noite']     = fuzz.trapmf(hora_cos.universe, [0.1, 0.4, 1.0, 1.0])

# --- HORA_SIN (Manhã vs Tarde) ---
# Manhã: > 0 | Tarde: < 0
hora_sin['tarde']     = fuzz.trapmf(hora_sin.universe, [-1.0, -1.0, -0.1, 0.0])
hora_sin['manha']     = fuzz.trapmf(hora_sin.universe, [0.0, 0.1, 1.0, 1.0])

# --- TIPO_NUVEM ---
# Sobreposição generosa para suavizar transições de nuvens
tipo_nuvem['limpo']     = fuzz.trimf(tipo_nuvem.universe, [0, 0, 4])
tipo_nuvem['parcial']   = fuzz.trimf(tipo_nuvem.universe, [2, 5, 8])
tipo_nuvem['encoberto'] = fuzz.trapmf(tipo_nuvem.universe, [6, 9, 10, 10])

# --- TEMPERATURA ---
temp_ar['conforto'] = fuzz.trapmf(temp_ar.universe, [10, 10, 20, 28])
temp_ar['quente']   = fuzz.trapmf(temp_ar.universe, [25, 32, 45, 45])


# --- GHI OUTPUT ---
ghi['zero']        = fuzz.trimf(ghi.universe, [0, 0, 10])
ghi['muito_baixo'] = fuzz.trimf(ghi.universe, [10, 150, 300])
ghi['baixo']       = fuzz.trimf(ghi.universe, [200, 350, 500])
ghi['medio']       = fuzz.trimf(ghi.universe, [400, 550, 700])
ghi['alto']        = fuzz.trimf(ghi.universe, [600, 750, 850])
ghi['muito_alto']  = fuzz.trimf(ghi.universe, [800, 900, 960])
ghi['extremo']     = fuzz.trimf(ghi.universe, [940, 1000, 1000])


# ======================================================
# 3. BASE DE REGRAS
# ======================================================

regras = []

# Noite ou Encoberto Total
regras.append(ctrl.Rule(hora_cos['noite'], ghi['zero']))
regras.append(ctrl.Rule(hora_cos['baixo'] & tipo_nuvem['encoberto'], ghi['zero']))

# Pico Limpo
regras.append(ctrl.Rule(hora_cos['zenite'] & tipo_nuvem['limpo'] & temp_ar['conforto'], ghi['extremo']))     # Frio = Eficiência Max
regras.append(ctrl.Rule(hora_cos['zenite'] & tipo_nuvem['limpo'] & temp_ar['quente'], ghi['muito_alto']))   # Calor = Perda leve

# Pico Parcial
regras.append(ctrl.Rule(hora_cos['zenite'] & tipo_nuvem['parcial'] & temp_ar['conforto'], ghi['alto']))     # Nuvens + Frio
regras.append(ctrl.Rule(hora_cos['zenite'] & tipo_nuvem['parcial'] & temp_ar['quente'], ghi['medio']))      # Nuvens + Calor

# Pico Encoberto
regras.append(ctrl.Rule(hora_cos['zenite'] & tipo_nuvem['encoberto'] & temp_ar['conforto'], ghi['baixo']))
regras.append(ctrl.Rule(hora_cos['zenite'] & tipo_nuvem['encoberto'] & temp_ar['quente'], ghi['muito_baixo']))

# Manhã Limpa (Geralmente a melhor hora depois do meio dia)
regras.append(ctrl.Rule(hora_cos['alto'] & hora_sin['manha'] & tipo_nuvem['limpo'] & temp_ar['conforto'], ghi['muito_alto']))
regras.append(ctrl.Rule(hora_cos['alto'] & hora_sin['manha'] & tipo_nuvem['limpo'] & temp_ar['quente'], ghi['alto']))

# Manhã Parcial
regras.append(ctrl.Rule(hora_cos['alto'] & hora_sin['manha'] & tipo_nuvem['parcial'] & temp_ar['conforto'], ghi['medio']))
regras.append(ctrl.Rule(hora_cos['alto'] & hora_sin['manha'] & tipo_nuvem['parcial'] & temp_ar['quente'], ghi['baixo']))

# Tarde Limpa (Sofre mais com calor/turbidez que a manhã)
regras.append(ctrl.Rule(hora_cos['alto'] & hora_sin['tarde'] & tipo_nuvem['limpo'] & temp_ar['conforto'], ghi['alto']))
regras.append(ctrl.Rule(hora_cos['alto'] & hora_sin['tarde'] & tipo_nuvem['limpo'] & temp_ar['quente'], ghi['medio']))

# Tarde Parcial
regras.append(ctrl.Rule(hora_cos['alto'] & hora_sin['tarde'] & tipo_nuvem['parcial'] & temp_ar['conforto'], ghi['baixo']))
regras.append(ctrl.Rule(hora_cos['alto'] & hora_sin['tarde'] & tipo_nuvem['parcial'] & temp_ar['quente'], ghi['muito_baixo']))

# Tarde Encoberto (e Manhã Encoberto - Catch All)
regras.append(ctrl.Rule(hora_cos['alto'] & tipo_nuvem['encoberto'], ghi['muito_baixo']))

# Nascer (Seno Manhã)
regras.append(ctrl.Rule(hora_cos['baixo'] & hora_sin['manha'] & (tipo_nuvem['limpo'] | tipo_nuvem['parcial']), ghi['baixo']))

# Pôr (Seno Tarde) - Rende menos
regras.append(ctrl.Rule(hora_cos['baixo'] & hora_sin['tarde'] & (tipo_nuvem['limpo'] | tipo_nuvem['parcial']), ghi['muito_baixo']))

# Compilação
controle_ghi = ctrl.ControlSystem(regras)
simulador = ctrl.ControlSystemSimulation(controle_ghi)


# ======================================================
# 4. INTERFACE
# ======================================================

def avaliar_ghi_mamdani(h_sin, h_cos, nuvem, temp):
    try:
        # Clipping rigoroso para evitar erros de limite
        simulador.input['hora_sin'] = np.clip(h_sin, -1, 1)
        simulador.input['hora_cos'] = np.clip(h_cos, -1, 1)
        simulador.input['tipo_nuvem'] = np.clip(nuvem, 0, 10)
        simulador.input['temp_ar'] = np.clip(temp, 10, 45)
        
        simulador.compute()
        return simulador.output['ghi']
    except:
        # Em caso de erro (ex: buraco nas regras), retorna 0 seguro
        return 0.0

def interpretar_resultado_ghi(valor):
    if valor < 50: return "Noite/Nulo"
    if valor < 300: return "Baixo"
    if valor < 600: return "Médio"
    if valor < 900: return "Alto"
    return "Extremo"