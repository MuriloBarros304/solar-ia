import numpy as np

# ======================================================
# LÓGICA SUGENO PARA GHI (CALIBRADA P/ ~920 W/m²)
# ======================================================

PESOS = {
    # --- CENÁRIOS DE MEIO-DIA ---
    
    # Céu Limpo @ Meio-dia
    # Antes: -450*(-1) + 600 = 1050 (Muito Alto)
    # Agora: -420*(-1) + 500 = 920 (Ideal)
    "pico_limpo": {
        "w": [0.0, -420.0, -20.0, 5.0],
        "b": 500.0
    },

    # Parcialmente Nublado @ Meio-dia
    # Meta: ~700 W/m²
    "pico_parcial": {
        "w": [0.0, -300.0, -20.0, 3.0],
        "b": 250.0
    },

    # Encoberto @ Meio-dia
    "pico_encoberto": {
        "w": [0.0, -100.0, -30.0, 1.0],
        "b": 50.0
    },

    # --- CENÁRIOS DE MANHÃ (h_sin > 0) ---

    # Manhã Limpa
    # Antes: Bias 650 era muito alto
    # Agora: Bias 550. Com cos -0.5 -> 225 + 550 = 775 (Mais realista)
    "dia_limpo_manha": {
        "w": [50.0, -450.0, -30.0, 4.0], 
        "b": 550.0 
    },

    "dia_parcial_manha": {
        "w": [30.0, -280.0, -40.0, 2.0],
        "b": 280.0
    },

    # --- CENÁRIOS DE TARDE (h_sin < 0) ---

    # Tarde Limpa
    # Reduzido levemente para refletir perda térmica
    "dia_limpo_tarde": {
        "w": [0.0, -420.0, -30.0, 4.0],
        "b": 500.0 
    },

    "dia_parcial_tarde": {
        "w": [0.0, -280.0, -40.0, 2.0],
        "b": 200.0
    },

    # Encoberto
    "dia_encoberto": {
        "w": [0.0, -80.0, -20.0, 0.5],
        "b": 40.0
    },

    # --- CENÁRIOS DE HORIZONTE ---

    "horizonte_nascer": {
        "w": [50.0, -300.0, -20.0, 1.0],
        "b": 200.0
    },

    "horizonte_por": {
        "w": [0.0, -300.0, -20.0, 1.0],
        "b": 150.0
    },

    "horizonte_nublado": {
        "w": [0.0, -80.0, -10.0, 0.0],
        "b": 30.0
    },

    # --- NOITE ---
    "noite": {
        "w": [0.0, 0.0, 0.0, 0.0],
        "b": 0.0 
    }
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
    
    # 2. Pertinências de PERÍODO
    p_periodo_manha = gaussian(h_sin, 0.7, 0.5)
    p_periodo_tarde = gaussian(h_sin, -0.7, 0.5)

    # Noite
    p_noite = 1.0 if h_cos > 0.2 else 0.0
    if h_cos > 0.1:
        p_horizonte *= (1 - (h_cos - 0.1)*10)
        p_horizonte = max(0, p_horizonte)

    # 3. NUVEM
    p_limpo     = gaussian(nuvem, 0.0, 2.0)
    p_parcial   = gaussian(nuvem, 5.0, 2.5)
    p_encoberto = gaussian(nuvem, 10.0, 2.5)

    regras = {}

    if p_noite > 0.5:
        return {"noite": 1.0}

    # Regras
    regras["pico_limpo"]     = p_meio_dia * p_limpo
    regras["pico_parcial"]   = p_meio_dia * p_parcial
    regras["pico_encoberto"] = p_meio_dia * p_encoberto

    regras["dia_limpo_manha"]   = p_manha_tarde * p_periodo_manha * p_limpo
    regras["dia_parcial_manha"] = p_manha_tarde * p_periodo_manha * p_parcial
    
    regras["dia_limpo_tarde"]   = p_manha_tarde * p_periodo_tarde * p_limpo
    regras["dia_parcial_tarde"] = p_manha_tarde * p_periodo_tarde * p_parcial
    
    regras["dia_encoberto"]     = p_manha_tarde * p_encoberto

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