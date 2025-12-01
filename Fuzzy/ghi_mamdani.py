import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ======================================================
# 1. DEFINIÇÃO DAS VARIÁVEIS
# ======================================================

hora_cos = ctrl.Antecedent(np.linspace(-1, 1, 500), 'hora_cos')
hora_sin = ctrl.Antecedent(np.linspace(-1, 1, 500), 'hora_sin')
tipo_nuvem = ctrl.Antecedent(np.linspace(0, 10, 200), 'tipo_nuvem')
temp_ar = ctrl.Antecedent(np.linspace(10, 45, 200), 'temp_ar')

# Output ajustado para corrigir o erro de centróide noturno
ghi = ctrl.Consequent(np.linspace(0, 1000, 1000), 'ghi', defuzzify_method='centroid')

# ======================================================
# 2. FUNÇÕES DE PERTINÊNCIA
# ======================================================

# --- HORA_COS ---
hora_cos['meio_dia']      = fuzz.trimf(hora_cos.universe, [-1, -1, -0.6])
hora_cos['dia_alto']      = fuzz.trimf(hora_cos.universe, [-0.8, -0.4, 0.0])
hora_cos['horizonte']     = fuzz.trimf(hora_cos.universe, [-0.2, 0.1, 0.4])
hora_cos['noite']         = fuzz.trapmf(hora_cos.universe, [0.2, 0.5, 1, 1])

# --- HORA_SIN ---
hora_sin['tarde']      = fuzz.trapmf(hora_sin.universe, [-1, -1, -0.2, 0.1])
hora_sin['transicao']  = fuzz.trimf(hora_sin.universe, [-0.3, 0, 0.3])
hora_sin['manha']      = fuzz.trapmf(hora_sin.universe, [-0.1, 0.2, 1, 1])

# --- TIPO_NUVEM ---
tipo_nuvem['limpo']       = fuzz.trimf(tipo_nuvem.universe, [0, 0, 3])
tipo_nuvem['parcial']     = fuzz.trimf(tipo_nuvem.universe, [2, 5, 8])
tipo_nuvem['encoberto']   = fuzz.trapmf(tipo_nuvem.universe, [6, 9, 10, 10])

# --- TEMPERATURA ---
temp_ar['baixa'] = fuzz.trapmf(temp_ar.universe, [10, 10, 20, 25])
temp_ar['alta']  = fuzz.trapmf(temp_ar.universe, [22, 28, 45, 45])

# --- GHI OUTPUT (CORREÇÃO DE BASE) ---
# CORREÇÃO CRUCIAL: Base reduzida de 50 para 5.
# O centróide de [0,0,50] é 16.66 (o erro da sua imagem).
# O centróide de [0,0,5] será ~1.6 (insignificante).
ghi['zero']        = fuzz.trimf(ghi.universe, [0, 0, 5]) 

ghi['muito_baixo'] = fuzz.trimf(ghi.universe, [5, 100, 200])
ghi['baixo']       = fuzz.trimf(ghi.universe, [150, 250, 350])
ghi['medio_baixo'] = fuzz.trimf(ghi.universe, [300, 400, 500])
ghi['medio']       = fuzz.trimf(ghi.universe, [450, 550, 650])
ghi['medio_alto']  = fuzz.trimf(ghi.universe, [600, 700, 800])
ghi['alto']        = fuzz.trimf(ghi.universe, [750, 850, 900])
ghi['extremo']     = fuzz.trimf(ghi.universe, [850, 950, 1000])

# ======================================================
# 3. REGRAS
# ======================================================

# --- BLOQUEIO TOTAL ---
r0 = ctrl.Rule(hora_cos['noite'], ghi['zero'])
r1 = ctrl.Rule(hora_cos['horizonte'] & tipo_nuvem['encoberto'], ghi['zero'])

# --- MEIO-DIA ---
r2 = ctrl.Rule(hora_cos['meio_dia'] & tipo_nuvem['limpo'], ghi['extremo'])
r3 = ctrl.Rule(hora_cos['meio_dia'] & tipo_nuvem['parcial'], ghi['medio_alto'])
r4 = ctrl.Rule(hora_cos['meio_dia'] & tipo_nuvem['encoberto'], ghi['baixo'])

# --- ASSIMETRIA MANHÃ vs TARDE ---
r5a = ctrl.Rule(hora_cos['dia_alto'] & hora_sin['manha'] & tipo_nuvem['limpo'], ghi['extremo'])
r5b = ctrl.Rule(hora_cos['dia_alto'] & hora_sin['tarde'] & tipo_nuvem['limpo'], ghi['alto'])

r6a = ctrl.Rule(hora_cos['dia_alto'] & hora_sin['manha'] & tipo_nuvem['parcial'], ghi['medio_alto'])
r6b = ctrl.Rule(hora_cos['dia_alto'] & hora_sin['tarde'] & tipo_nuvem['parcial'], ghi['medio'])

r7 = ctrl.Rule(hora_cos['dia_alto'] & tipo_nuvem['encoberto'], ghi['muito_baixo'])

# --- HORIZONTE ---
r8a = ctrl.Rule(hora_cos['horizonte'] & hora_sin['manha'] & tipo_nuvem['limpo'], ghi['baixo'])
r8b = ctrl.Rule(hora_cos['horizonte'] & hora_sin['tarde'] & tipo_nuvem['limpo'], ghi['muito_baixo'])

r10 = ctrl.Rule(hora_cos['meio_dia'] & temp_ar['alta'], ghi['extremo'])

controle_ghi = ctrl.ControlSystem([r0, r1, r2, r3, r4, r5a, r5b, r6a, r6b, r7, r8a, r8b, r10])
simulador = ctrl.ControlSystemSimulation(controle_ghi)

# ======================================================
# 4. INTERFACE
# ======================================================

def avaliar_ghi_mamdani(h_sin, h_cos, nuvem, temp):
    try:
        simulador.input['hora_sin'] = np.clip(h_sin, -1, 1)
        simulador.input['hora_cos'] = np.clip(h_cos, -1, 1)
        simulador.input['tipo_nuvem'] = np.clip(nuvem, 0, 10)
        simulador.input['temp_ar'] = np.clip(temp, 10, 45)
        simulador.compute()
        return simulador.output['ghi']
    except:
        return 0.0

def Interpretar_resultado_ghi(valor):
    if valor < 50: return "Noite/Nulo"
    if valor < 300: return "Baixo"
    if valor < 600: return "Médio"
    return "Alto"