import google.generativeai as genai
import streamlit as st
import os
import re

IMAGE_MAP = {
    # Palavras-chave genéricas
    "comparação": "predict/comparacao_series_temporais.png",
    "dispersão": "predict/analise_dispersao_erro.png",
    
    # Palavras-chave específicas do modelo
    "importância rf": "predict/feature_importance_randomforest.png",
    "importância xg": "predict/feature_importance_xgboost_(ghi_model).png",
    "predição diária xg": "predict/predicao_diaria_xg.png"
}

def load_context():
    """Carrega o arquivo de texto de contextualização."""
    try:
        with open("context.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        st.error("Arquivo 'contexto_projeto.txt' não encontrado.")
        return "" # Retorna um contexto vazio se o arquivo não for encontrado

def find_images_in_response(response_text):
    """
    Verifica o texto de resposta por palavras-chave e retorna
    os caminhos das imagens correspondentes.
    """
    images_to_show = set()
    text_lower = response_text.lower()
    
    for keyword, path in IMAGE_MAP.items():
        # Usamos regex com \b para encontrar palavras inteiras
        if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower, re.IGNORECASE):
            images_to_show.add(path)
            
    return list(images_to_show)

def run_ai(user_prompt):
    """
    Executa a consulta à API do Gemini, combinando o contexto e o prompt do usuário.
    Retorna o texto da resposta e uma lista de imagens para exibir.
    """
    context_text = load_context()
    if not context_text:
        return "Erro: Não foi possível carregar o contexto do projeto.", []

    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key) # type: ignore
    except KeyError:
        return "Erro: A GEMINI_API_KEY não foi configurada nos Segredos (Secrets) do Streamlit.", []
    except Exception as e:
        return f"Erro ao configurar a API do Gemini: {e}", []

    full_prompt = f"""
    {context_text}
    
    ---
    
    ENTRADA DO USUÁRIO:
    {user_prompt}
    
    ---
    
    Com base estritamente no contexto fornecido, responda à pergunta do usuário.
    Se a resposta exigir a visualização de um gráfico, mencione palavras-chave
    como 'comparação', 'dispersão', 'importância', ou 'predição diária'
    em sua resposta para que a imagem correta possa ser exibida.
    """
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash') # type: ignore
        response = model.generate_content(full_prompt)
        response_text = response.text
    except Exception as e:
        return f"Erro ao gerar resposta da IA: {e}", []

    images = find_images_in_response(response_text)
    
    return response_text, images