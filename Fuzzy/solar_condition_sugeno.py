import numpy as np

# Modelo Sugeno baseado em combinação linear

# Regras: saída = w1*x1 + w2*x2 + w3*x3 + w4*x4 + b

PESOS = {
    "regra1":  {"w": [2.0, 1.5, -0.3, 0.2], "b": 3.0},
    "regra2":  {"w": [1.0, 2.0, -0.5, 0.1], "b": 2.0},
    "regra3":  {"w": [0.5, 1.0, -1.0, 0.0], "b": 1.0},
    "regra4":  {"w": [2.5, 1.0, -0.2, 0.3], "b": 2.5},
    "regra5":  {"w": [1.5, 1.5, -0.4, 0.2], "b": 2.0},
}

# Grau de ativação de cada regra (Simples: média ponderada)
def ativacao(h_sin, h_cos, nuvem, temp):
    return {
        "regra1": np.clip(1 - abs(nuvem - 2)/5, 0, 1),
        "regra2": np.clip(1 - abs(temp - 25)/10, 0, 1),
        "regra3": np.clip(1 - nuvem/8, 0, 1),
        "regra4": np.clip((h_sin + 1)/2, 0, 1),
        "regra5": np.clip((h_cos + 1)/2, 0, 1),
    }

# Avaliação Sugeno
def avaliar_condicao_solar_sugeno(h_sin, h_cos, nuvem, temp):
    entradas = np.array([h_sin, h_cos, nuvem, temp])
    a = ativacao(h_sin, h_cos, nuvem, temp)

    num = 0
    den = 0

    for chave, grau in a.items():
        w = np.array(PESOS[chave]["w"])
        b = PESOS[chave]["b"]
        saida_regra = np.dot(w, entradas) + b
        num += grau * saida_regra
        den += grau

    return num / den if den != 0 else 0