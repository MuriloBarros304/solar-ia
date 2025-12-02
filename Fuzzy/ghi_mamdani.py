import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ======================================================
# 1. DEFINIÇÃO DAS VARIÁVEIS (REFINAMENTO ESTRUTURAL)
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
# Universo estendido para 1200 para permitir que o centróide atinja picos de 1000+
ghi = ctrl.Consequent(np.linspace(0, 1000, 1000), 'ghi', defuzzify_method='centroid')


# ======================================================
# 2. FUNÇÕES DE PERTINÊNCIA (ALTA SIMETRIA E OVERLAP)
# ======================================================
# A chave para o Mamdani funcionar bem é a sobreposição (overlap).
# Se um conjunto termina onde o outro começa, cria-se "buracos" na lógica.
# Aqui usamos sobreposição de ~50%.

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
# 3. BASE DE REGRAS (MATRIZ LÓGICA COMPLETA)
# ======================================================

regras = []

# --- GRUPO 1: BLOQUEIO TOTAL (Prioridade Máxima) ---
regras.append(ctrl.Rule(hora_cos['noite'], ghi['zero']))
regras.append(ctrl.Rule(hora_cos['baixo'] & tipo_nuvem['encoberto'], ghi['zero']))

# --- GRUPO 2: ZÊNITE (MEIO-DIA) ---
# Sol a pino (-1). Potencial máximo.
# Limpo
regras.append(ctrl.Rule(hora_cos['zenite'] & tipo_nuvem['limpo'] & temp_ar['conforto'], ghi['extremo'])) # Frio ajuda
regras.append(ctrl.Rule(hora_cos['zenite'] & tipo_nuvem['limpo'] & temp_ar['quente'], ghi['muito_alto'])) # Calor atrapalha um pouco

# Parcial
regras.append(ctrl.Rule(hora_cos['zenite'] & tipo_nuvem['parcial'], ghi['alto']))

# Encoberto
regras.append(ctrl.Rule(hora_cos['zenite'] & tipo_nuvem['encoberto'], ghi['baixo']))

# --- GRUPO 3: SOL ALTO (MANHÃ/TARDE - ~45°) ---
# Aqui entra a assimetria do Seno (Manhã rende mais que Tarde)

# Manhã (Painel frio, atmosfera limpa)
regras.append(ctrl.Rule(hora_cos['alto'] & hora_sin['manha'] & tipo_nuvem['limpo'], ghi['muito_alto']))
regras.append(ctrl.Rule(hora_cos['alto'] & hora_sin['manha'] & tipo_nuvem['parcial'], ghi['medio']))

# Tarde (Painel quente, turbulência) - Penalidade leve
regras.append(ctrl.Rule(hora_cos['alto'] & hora_sin['tarde'] & tipo_nuvem['limpo'], ghi['alto']))
regras.append(ctrl.Rule(hora_cos['alto'] & hora_sin['tarde'] & tipo_nuvem['parcial'], ghi['medio'])) # Mantém médio, mas o centróide cairá um pouco pela sobreposição

# Encoberto (Indifere horário)
regras.append(ctrl.Rule(hora_cos['alto'] & tipo_nuvem['encoberto'], ghi['muito_baixo']))

# --- GRUPO 4: SOL BAIXO (HORIZONTE) ---
# Nascer/Pôr do sol. A atmosfera filtra muito a luz.

# Limpo
regras.append(ctrl.Rule(hora_cos['baixo'] & tipo_nuvem['limpo'], ghi['baixo']))

# Parcial
regras.append(ctrl.Rule(hora_cos['baixo'] & tipo_nuvem['parcial'], ghi['muito_baixo']))

# --- GRUPO 5: CONDIÇÕES EXTREMAS (REFORÇO) ---
# Se estiver muito limpo e for meio dia, força Extremo independente da temperatura (para garantir picos)
regras.append(ctrl.Rule(hora_cos['zenite'] & tipo_nuvem['limpo'], ghi['extremo']))

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