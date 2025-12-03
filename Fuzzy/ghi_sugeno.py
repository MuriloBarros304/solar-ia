import numpy as np

# ======================================================
# LÓGICA SUGENO PARA GHI
# ======================================================

PESOS = {
    # Equações no formato:
    # y = w1*x1 + w2*x2 + w3*x3 + w4*x4 + b
    # onde x1, x2, x3 e x4 são as entradas: [hora_sin, hora_cos, tipo_nuvem, temp_ar]
    # b é o bias (intercepto)
    # --- CENÁRIOS DE MEIO-DIA (PICO) ---
    # Pico Limpo + Frio (Eficiência Máxima) -> Meta: ~950 W/m²
    "pico_limpo_frio": {
        "w": [0.0, -450.0, -20.0, 6.0], 
        "b": 550.0
    },
    # Pico Limpo + Calor (Perda Térmica) -> Meta: ~880 W/m²
    "pico_limpo_calor": {
        "w": [0.0, -400.0, -20.0, 2.0], 
        "b": 480.0
    },

    # Pico Parcial + Frio -> Meta: ~750 W/m²
    "pico_parcial_frio": {
        "w": [0.0, -320.0, -20.0, 4.0],
        "b": 300.0
    },
    # Pico Parcial + Calor -> Meta: ~680 W/m²
    "pico_parcial_calor": {
        "w": [0.0, -280.0, -20.0, 2.0],
        "b": 240.0
    },

    # Pico Encoberto (Diferença térmica é pequena aqui)
    "pico_encoberto_frio": { "w": [0.0, -100.0, -30.0, 1.0], "b": 60.0 },
    "pico_encoberto_calor": { "w": [0.0, -100.0, -30.0, 0.5], "b": 40.0 },


    # --- CENÁRIOS DE MANHÃ (Sol Leste) ---
    
    # Manhã Limpa + Frio (Cenário Ideal Matinal) -> Meta: ~800 W/m²
    "dia_limpo_manha_frio": {
        "w": [60.0, -450.0, -30.0, 5.0], 
        "b": 600.0 
    },
    # Manhã Limpa + Calor -> Meta: ~750 W/m²
    "dia_limpo_manha_calor": {
        "w": [40.0, -400.0, -30.0, 2.0],
        "b": 520.0
    },

    # Manhã Parcial
    "dia_parcial_manha_frio": { "w": [40.0, -300.0, -40.0, 3.0], "b": 320.0 },
    "dia_parcial_manha_calor": { "w": [30.0, -260.0, -40.0, 1.0], "b": 260.0 },


    # --- CENÁRIOS DE TARDE (Sol Oeste) ---
    # Tardes geralmente são quentes, Frio aqui seria "frente fria"
    
    # Tarde Limpa + Frio
    "dia_limpo_tarde_frio": {
        "w": [0.0, -420.0, -30.0, 4.0],
        "b": 550.0 
    },
    
    # Tarde Parcial + Frio
    "dia_parcial_tarde_frio": { "w": [0.0, -300.0, -40.0, 2.0], "b": 250.0 },
    
    # Encoberto Tarde Frio
    "dia_encoberto_tarde_frio": { "w": [0.0, -80.0, -20.0, 0.5], "b": 50.0 },

    # Fallback Geral para Encoberto/Outros
    "dia_encoberto": { "w": [0.0, -80.0, -20.0, 0.5], "b": 40.0 },


    # --- CENÁRIOS DE HORIZONTE ---
    "horizonte_nascer":  { "w": [50.0, -300.0, -20.0, 1.0], "b": 200.0 },
    "horizonte_por":     { "w": [0.0, -300.0, -20.0, 1.0], "b": 150.0 },
    "horizonte_nublado": { "w": [0.0, -80.0, -10.0, 0.0], "b": 30.0 },

    # --- NOITE ---
    "noite": { "w": [0.0, 0.0, 0.0, 0.0], "b": 0.0 }
}

# ======================================================
# FUNÇÕES DE ATIVAÇÃO
# ======================================================

def gaussian(x, mean, sigma):
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2)

def calcular_ativacao(h_sin, h_cos, nuvem, temp):
    # 1. Pertinências de HORA
    p_meio_dia    = gaussian(h_cos, -1.0, 0.3)
    p_manha_tarde = gaussian(h_cos, -0.5, 0.3)
    p_horizonte   = gaussian(h_cos, 0.0, 0.2)
    
    # 2. Pertinências de PERÍODO (Seno)
    p_periodo_manha = gaussian(h_sin, 0.7, 0.5)
    p_periodo_tarde = gaussian(h_sin, -0.7, 0.5)

    # 3. Noite
    p_noite = 1.0 if h_cos > 0.2 else 0.0
    if h_cos > 0.1:
        p_horizonte *= (1 - (h_cos - 0.1)*10)
        p_horizonte = max(0, p_horizonte)

    # 4. Pertinências de NUVEM
    p_limpo     = gaussian(nuvem, 0.0, 2.0)
    p_parcial   = gaussian(nuvem, 5.0, 2.5)
    p_encoberto = gaussian(nuvem, 10.0, 2.5)

    # 5. Pertinências de TEMPERATURA
    # Frio: < 20°C (Favorece PV) | Calor: > 30°C (Desfavorece PV)
    p_frio  = gaussian(temp, 20.0, 10.0)
    p_calor = gaussian(temp, 35.0, 10.0)

    regras = {}

    if p_noite > 0.5:
        return {"noite": 1.0}

    # --- REGRAS ---
    
    # PICO (Meio-dia)
    regras["pico_limpo_frio"]     = p_meio_dia * p_limpo * p_frio
    regras["pico_limpo_calor"]    = p_meio_dia * p_limpo * p_calor
    
    regras["pico_parcial_frio"]   = p_meio_dia * p_parcial * p_frio
    regras["pico_parcial_calor"]  = p_meio_dia * p_parcial * p_calor
    
    regras["pico_encoberto_frio"] = p_meio_dia * p_encoberto * p_frio
    regras["pico_encoberto_calor"]= p_meio_dia * p_encoberto * p_calor

    # MANHÃ
    regras["dia_limpo_manha_frio"]      = p_manha_tarde * p_periodo_manha * p_limpo * p_frio
    regras["dia_limpo_manha_calor"]     = p_manha_tarde * p_periodo_manha * p_limpo * p_calor
    
    regras["dia_parcial_manha_frio"]    = p_manha_tarde * p_periodo_manha * p_parcial * p_frio
    regras["dia_parcial_manha_calor"]   = p_manha_tarde * p_periodo_manha * p_parcial * p_calor

    # TARDE
    regras["dia_limpo_tarde_frio"]      = p_manha_tarde * p_periodo_tarde * p_limpo * p_frio
    regras["dia_parcial_tarde_frio"]    = p_manha_tarde * p_periodo_tarde * p_parcial * p_frio
    regras["dia_encoberto_tarde_frio"]  = p_manha_tarde * p_periodo_tarde * p_encoberto * p_frio

    # Catch-all para Encoberto (Manhã/Tarde calor ou geral)
    regras["dia_encoberto"]             = p_manha_tarde * p_encoberto * max(p_calor, 0.2)

    # HORIZONTE
    regras["horizonte_nascer"]  = p_horizonte * p_periodo_manha * max(p_limpo, p_parcial)
    regras["horizonte_por"]     = p_horizonte * p_periodo_tarde * max(p_limpo, p_parcial)
    regras["horizonte_nublado"] = p_horizonte * p_encoberto

    return regras

# ======================================================
# AVALIAÇÃO DO MODELO
# ======================================================

def avaliar_ghi_sugeno(h_sin, h_cos, nuvem, temp):
    h_cos = np.clip(h_cos, -1, 1)
    h_sin = np.clip(h_sin, -1, 1)
    nuvem = np.clip(nuvem, 0, 10)
    temp = np.clip(temp, 10, 45)
    
    entradas = np.array([h_sin, h_cos, nuvem, temp])
    ativacoes = calcular_ativacao(h_sin, h_cos, nuvem, temp)

    numerador = 0.0
    denominador = 0.0

    for regra, grau in ativacoes.items():
        if grau > 0.001:
            if regra in PESOS:
                coefs = PESOS[regra]
                w = np.array(coefs["w"])
                b = coefs["b"]
                y_regra = np.dot(w, entradas) + b
                numerador += grau * y_regra
                denominador += grau

    if denominador == 0:
        return 0.0

    ghi_estimado = numerador / denominador
    return float(np.clip(ghi_estimado, 0, 1400))

def interpretar_ghi_sugeno(valor):
    if valor < 50: return "Noite/Nulo"
    if valor < 400: return "Baixo"
    if valor < 800: return "Médio"
    return "Alto"