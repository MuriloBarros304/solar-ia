import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
import os
import chat
from datetime import datetime, time
from Fuzzy.solar_condition import avaliar_condicao_solar,interpretar_condicao_fuzzy, hora_sin, hora_cos, tipo_nuvem, temp_ar, condicao_solar, plot_var
from streamlit_folium import st_folium

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --- 1. CONFIGURAÇÃO DA PÁGINA E CARREGAMENTO DE MODELOS ---
st.set_page_config(
    page_title="Irradiação Solar no RN",
    page_icon="☀️",
    layout="wide"
)

@st.cache_resource
def load_model_and_features():
    """Carrega o modelo RandomForest e a lista de nomes das features."""
    try:
        # model = joblib.load('training/random_forest_model.joblib')
        model_ghi = joblib.load('training/xgb_model_ghi.joblib')
        model_dni = joblib.load('training/xgb_model_dni.joblib')
        features = joblib.load('training/model_features.joblib')
        return model_dni, model_ghi, features
    except FileNotFoundError:
        st.error(f"ERRO CRÍTICO: Arquivos do modelo (xgb_model_ghi.joblib ou xgb_model_dni.joblib ou 'random_forest_model.joblib' ou 'model_features.joblib') não encontrados na pasta 'training/'. Execute os scripts de treino primeiro.")
        return None, None, None

@st.cache_data
def load_validation_data():
    """Carrega os dados de validação para obter a lista de features e para comparações."""
    try:
        X_val = pd.read_parquet('data/X_val.parquet', engine='fastparquet')
        y_val = pd.read_parquet('data/y_val.parquet', engine='fastparquet')
        df_full = pd.read_parquet('data/dataframe.parquet', engine='fastparquet')
        val_df_full = df_full[(df_full.index >= '2023-01-01') & (df_full.index < '2024-01-01')]
        
        return X_val, y_val, val_df_full # Retorna os três DataFrames
    except FileNotFoundError:
        st.error("ERRO CRÍTICO: Arquivos de dados (X_val.parquet, y_val.parquet) não encontrados na pasta `data/`. Execute o script 'dataframe.py' primeiro.")
        return None, None, None

X_val, y_val, val_df_full = load_validation_data()
# model, feature_names = load_model_and_features()
model_dni, model_ghi, feature_names = load_model_and_features()
# if model is None or feature_names is None:
#     st.stop()
if model_dni is None or model_ghi is None or feature_names is None:
    st.stop()

if 'lat' not in st.session_state:
    st.session_state.lat = -5.7944 # Ponto inicial do marcador
    st.session_state.lon = -36.5667 # Ponto inicial do marcador
    st.session_state.map_center = [-5.7944, -36.5667] # Centro inicial do mapa

st.sidebar.title("🛰️ Painel de Controle Global")
st.sidebar.header("Localização Geográfica")
st.sidebar.markdown("Selecione no mapa na primeira aba ou ajuste manualmente aqui.")

# Widgets para o usuário digitar. O valor default vem do session_state.
lat = st.sidebar.number_input("Latitude", value=st.session_state.lat, format="%.4f")
lon = st.sidebar.number_input("Longitude", value=st.session_state.lon, format="%.4f")

st.sidebar.header("Condições Meteorológicas Base")
st.sidebar.markdown("Valores usados como padrão nas simulações.")
user_wind_speed = st.sidebar.slider("Velocidade do Vento (m/s)", 0.0, 20.0, 5.0)
user_wind_dir = st.sidebar.slider("Direção do Vento (°)", 0, 360, 130)
user_cloud_type = st.sidebar.slider(
    "Tipo de Nuvem", 0, 8, 1,
    help="Um valor de 0 representa céu limpo, enquanto 8 representa céu totalmente encoberto (escala em Oktas)."
)

st.title("☀️ Análise de Irradiação Solar no RN")

tab_mapa, tab_pontual, tab_diaria, tab_anual, tab_chat, aba_fuzzy = st.tabs([
    "Mapa Interativo", "Regressão Pontual", "Regressão Diária", "Análise Anual", "Chatbot", "Sistema Fuzzy"
])

# ==============================================================================
# ABA 1: MAPA INTERATIVO
# ==============================================================================
with tab_mapa:
    st.header("Selecione a Coordenada no Mapa do RN")
    st.markdown("Clique em qualquer ponto dentro da área destacada para definir a latitude e a longitude que serão usadas em todas as simulações.")

    # Coordenadas aproximadas dos pontos extremos do estado
    RN_BOUNDS = {
        "lat_min": -6.98, "lat_max": -4.82,
        "lon_min": -38.58, "lon_max": -34.98
    }
    
    # Lógica de sincronização com a sidebar
    if lat != st.session_state.lat:
        st.session_state.lat = lat
    if lon != st.session_state.lon:
        st.session_state.lon = lon
    if 'df_plot_diaria' not in st.session_state:
        st.session_state.df_plot_diaria = pd.DataFrame()
    if 'real_features_diaria' not in st.session_state:
        st.session_state.real_features_diaria = None
    if 'real_target_diaria' not in st.session_state:
        st.session_state.real_target_diaria = None
    if 'last_sim_date' not in st.session_state:
        st.session_state.last_sim_date = None

    # Cria o objeto do mapa com Folium
    m = folium.Map(location=st.session_state.map_center, zoom_start=8, min_lat=-6.98, max_lat=-4.82, min_lon=-38.58, max_lon=-34.98, min_zoom=7, max_zoom=14)
    
    folium.Rectangle(
        bounds=[[RN_BOUNDS["lat_min"], RN_BOUNDS["lon_min"]], [RN_BOUNDS["lat_max"], RN_BOUNDS["lon_max"]]],
        color='#ff7800',
        fill=True,
        fill_color='#ffff00',
        fill_opacity=0.1,
        tooltip="Área de cobertura do modelo"
    ).add_to(m)

    # Adiciona um marcador na localização ATUAL do session_state
    folium.Marker(
        [st.session_state.lat, st.session_state.lon], 
        popup=f"Lat: {st.session_state.lat:.4f}, Lon: {st.session_state.lon:.4f}", 
        tooltip="Localização Selecionada"
    ).add_to(m)

    # Renderiza o mapa e captura o output
    map_data = st_folium(m, height=450, width=700)
    
    # Verifica se o usuário clicou no mapa
    if map_data and map_data.get("last_clicked"):
        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lon = map_data["last_clicked"]["lng"]
        
        is_lat_valid = RN_BOUNDS["lat_min"] <= clicked_lat <= RN_BOUNDS["lat_max"]
        is_lon_valid = RN_BOUNDS["lon_min"] <= clicked_lon <= RN_BOUNDS["lon_max"]

        if is_lat_valid and is_lon_valid:
            # Se o local clicado for diferente do que está salvo, atualiza e re-executa
            if st.session_state.lat != clicked_lat or st.session_state.lon != clicked_lon:
                st.session_state.lat = clicked_lat
                st.session_state.lon = clicked_lon
                st.rerun()
        else:
            # Se o clique for fora da área, exibe um aviso
            st.warning("📍 Ponto fora da área de cobertura. Por favor, clique dentro dos limites do Rio Grande do Norte.")
    st.header("Sobre o projeto")
    st.markdown("""
    Esta plataforma foi desenvolvida para fornecer previsões precisas de irradiação geral horizontal (GHI) e irradiação direta normal (DNI) no estado do Rio Grande do Norte, Brasil. Utilizando um modelo de XGBoost treinado com dados históricos de estações meteorológicas locais, o sistema permite simulações personalizadas com base em condições climáticas específicas e localização geográfica.
    """)

# ==============================================================================
# ABA 2: PREVISÃO PONTUAL
# ==============================================================================
with tab_pontual:
    st.header("Predição para uma Hora Específica")
    st.markdown("Ajuste os parâmetros geográficos e meteorológicos na **barra lateral** e as condições de tempo abaixo para ver a previsão ser atualizada em tempo real.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Parâmetros Temporais e Climáticos")
        
        c1, c2 = st.columns(2)
        with c1:
            data = st.date_input("Data da Predição", datetime(2023, 10, 29), format="DD/MM/YYYY")
        with c2:
            # Substituímos o time_input por um slider de hora, restrito ao período diurno
            hora_int = st.slider("Hora da Predição (HH)",
                                 min_value=7,
                                 max_value=17,
                                 value=12, # Padrão
                                 step=1,
                                 format="%02d") # Formato de exibição
            minute_int = st.slider("Minutos da Predição (MM)",
                                   min_value=0,
                                   max_value=55,
                                   value=0, # Padrão
                                   step=5,
                                   format="%02d") # Formato de exibição

        # Converte a hora (int) de volta para um objeto 'time' para o resto do script
        hora = time(hour=hora_int, minute=minute_int)

        with st.expander("Ajustar Parâmetros Climáticos"):
            temp_ar = st.slider("Temperatura do Ar (°C)", 18.0, 40.0, 28.0)
            umidade_rel = st.slider("Umidade Relativa (%)", 10.0, 100.0, 70.0)
            pressao_atm = st.slider("Pressão Atmosférica Estimada (mB)", 980.0, 1050.0, 1010.0, key="pressao_pontual")
            precipitacao = st.slider("Precipitação (mm)", 0.0, 50.0, 0.0)

    timestamp = datetime.combine(data, hora)

    # Construção do dicionário de input
    input_data = {}
    input_data['hora_sin'] = np.sin(2 * np.pi * timestamp.hour / 24.0)
    input_data['hora_cos'] = np.cos(2 * np.pi * timestamp.hour / 24.0)
    input_data['dia_ano_sin'] = np.sin(2 * np.pi * timestamp.timetuple().tm_yday / 365.25)
    input_data['dia_ano_cos'] = np.cos(2 * np.pi * timestamp.timetuple().tm_yday / 365.25)

    input_data.update({
        'latitude_inmet': lat,
        'longitude_inmet': lon,
        'temp_ar': temp_ar,
        'umidade_rel': umidade_rel,
        'pressao_atm_estacao': pressao_atm,
        'vento_vel': user_wind_speed,
        'vento_dir': user_wind_dir,
        'precipitacao': precipitacao,
        'tipo_nuvem': user_cloud_type,
    })

    input_df = pd.DataFrame([input_data])[feature_names]

    # Random Forest
    # ghi_prediction = model.predict(input_df)[0][1] # GHI
    # dni_prediction = model.predict(input_df)[0][0] # DNI

    # XGBoost
    ghi_prediction = model_ghi.predict(input_df)[0] # GHI
    dni_prediction = model_dni.predict(input_df)[0] # DNI

    with col2:
        st.subheader("Resultado da Predição")
        metric1, metric2 = st.columns(2)
        with metric1:
            st.metric(label="GHI Predito", value=f"{ghi_prediction:.2f} W/m²")
        with metric2:
            st.metric(label="DNI Predito", value=f"{dni_prediction:.2f} W/m²")

        with st.expander("Ver Detalhes da Condição Solar (Lógica Fuzzy)"):
            try:
                condicao_valor = avaliar_condicao_solar(ghi_prediction)
                mostrar_grafico_condicao_solar(ghi_prediction)  # mostra o gráfico fuzzy
                
                descricao = interpretar_condicao_fuzzy(condicao_valor)

                if condicao_valor < 40:
                     st.warning(descricao)
                elif condicao_valor < 70:
                      st.info(descricao)
                else:
                    st.success(descricao)
            except Exception as e:
                st.error(f"Erro ao avaliar condição solar: {e}")
   
    # O explorador de dados reais continua fora do expander, o que está correto.
    st.markdown("---")
    st.subheader("Explorar Dados Reais (2023)")
    # ... (o resto do seu código permanece igual) ...
    st.markdown(
        "- A304 - Natal\n"
        "- A316 - Caicó\n"
        "- A372 - Macau\n"
        "- A340 - Apodi\n"
    )

    if data.year == 2023 and val_df_full is not None:
        try:
            selected_timestamp = pd.to_datetime(timestamp.replace(minute=0, second=0, microsecond=0))
            real_data_at_hour = val_df_full[val_df_full.index == selected_timestamp]

            if not real_data_at_hour.empty:
                station_list = real_data_at_hour['codigo_estacao'].unique()
                selected_station = st.selectbox(
                    "Selecione uma Estação para ver os dados reais:",
                    station_list,
                    key="pontual_station_select"
                )
                
                if selected_station:
                    station_data = real_data_at_hour[
                        real_data_at_hour['codigo_estacao'] == selected_station
                    ].iloc[0]
                    
                    st.write(f"Dados reais para Estação **{selected_station}** às **{hora.strftime('%H:%M')}**:")
                    c1_real, c2_real, c3_real = st.columns(3)
                    c1_real.metric(label="GHI Real (W/m²)", value=f"{station_data.get('ghi', 'N/A'):.2f}")
                    c1_real.metric(label="Temp. Ar (°C)", value=f"{station_data.get('temp_ar', 'N/A'):.2f}")
                    c2_real.metric(label="DNI Real (W/m²)", value=f"{station_data.get('dni', 'N/A'):.2f}")
                    c2_real.metric(label="Umidade Rel. (%)", value=f"{station_data.get('umidade_rel', 'N/A'):.2f}")
                    c3_real.metric(label="Pressão (mB)", value=f"{station_data.get('pressao_atm_estacao', 'N/A'):.1f}")
                    c3_real.metric(label="Tipo de Nuvem", value=f"{station_data.get('tipo_nuvem', 'N/A'):.0f}")
            else:
                st.info("Não há dados reais de 2023 disponíveis para esta data e hora.")
        
        except Exception as e:
            st.error(f"Erro ao processar dados reais: {e}")
    
    elif data.year != 2023:
        st.info("Selecione uma data em 2023 para habilitar o explorador de dados reais.")

# ==============================================================================
# ABA 3: PREVISÃO DIÁRIA
# ==============================================================================
with tab_diaria:
    st.header("Simulação da Curva de GHI e DNI para um Dia Inteiro")
    st.markdown("Use os parâmetros geográficos e meteorológicos da **barra lateral** e defina a temperatura e umidade ao meio-dia para gerar a previsão hora a hora.")

    col_dia_1, col_dia_2 = st.columns([1, 2])
    
    with col_dia_1:
        data_simulacao = st.date_input("Data para Simulação", datetime(2023, 10, 29), key="sim_date", format="DD/MM/YYYY")
        temp_meio_dia = st.slider("Temp. ao Meio-Dia (°C)", 25.0, 42.0, 32.0)
        umidade_meio_dia = st.slider("Umidade ao Meio-Dia (%)", 20.0, 100.0, 60.0)
        precipitacao_dia = st.slider("Precipitação Diária (mm)", 0.0, 50.0, 0.0)
        pressao_atm_dia = st.slider("Pressão Atmosférica Estimada (mB)", 980.0, 1050.0, 1010.0)
        target_radio = st.radio("Selecione o alvo da previsão:", ('GHI', 'DNI'), index=0, key='target_radio')

        # --- LÓGICA DE PREPARAÇÃO DE DADOS REAIS ---
        if data_simulacao != st.session_state.last_sim_date:
            st.session_state.last_sim_date = data_simulacao
            st.session_state.real_features_diaria = None  # Reseta os dados
            st.session_state.real_target_diaria = None  # Reseta os dados
            
            if y_val is not None and data_simulacao.year == 2023:
                start_date = pd.to_datetime(data_simulacao)
                end_date = start_date + pd.Timedelta(days=1)

                # 1. Prepara dados de TARGET (GHI/DNI) reais
                real_data_day_target = y_val[(y_val.index >= start_date) & (y_val.index < end_date)]
                if not real_data_day_target.empty:
                    st.session_state.real_target_diaria = real_data_day_target.resample('h').mean().rename(columns={
                        'ghi': 'GHI Real', 'dni': 'DNI Real'
                    })

                # 2. Prepara dados de FEATURES reais
                if X_val is not None:
                    real_data_day_features = X_val[(X_val.index >= start_date) & (X_val.index < end_date)]
                    if not real_data_day_features.empty:
                        st.session_state.real_features_diaria = real_data_day_features.resample('h').mean()
                else:
                    # Se não há dados de validação carregados, garante que não há erro e mantém None
                    st.session_state.real_features_diaria = None
            
            # Limpa o gráfico de simulação antigo se a data mudar
            st.session_state.df_plot_diaria = pd.DataFrame()


        if st.button("Gerar Previsão Diária", type="primary"):
            # O botão agora SÓ gera a simulação e salva no session_state
            
            def simulate_hourly_variation(daily_value, peak_hour=12, min_factor=0.8):
                hourly_values = [daily_value * (min_factor + (1 - min_factor) * max(0, np.sin((h - (peak_hour - 6)) * np.pi / 12))) for h in range(24)]
                return hourly_values

            temp_horaria = simulate_hourly_variation(temp_meio_dia)
            umidade_horaria = simulate_hourly_variation(umidade_meio_dia, peak_hour=5, min_factor=0.9)
            timestamps_dia = [datetime.combine(data_simulacao, time(h)) for h in range(24)]
            predictions_dia = []

            with st.spinner("Gerando previsão hora a hora..."):
                for h, timestamp_hora in enumerate(timestamps_dia):
                    input_data_hora = {
                        'hora_sin': np.sin(2 * np.pi * h / 24.0), 'hora_cos': np.cos(2 * np.pi * h / 24.0),
                        'dia_ano_sin': np.sin(2 * np.pi * timestamp_hora.timetuple().tm_yday / 365.25),
                        'dia_ano_cos': np.cos(2 * np.pi * timestamp_hora.timetuple().tm_yday / 365.25),
                        'latitude_inmet': lat, 'longitude_inmet': lon, 'temp_ar': temp_horaria[h],
                        'umidade_rel': umidade_horaria[h], 'pressao_atm_estacao': pressao_atm_dia,
                        'vento_vel': user_wind_speed, 'vento_dir': user_wind_dir,
                        'precipitacao': precipitacao_dia, 'tipo_nuvem': user_cloud_type,
                    }
                    input_df_hora = pd.DataFrame([input_data_hora])[feature_names]
                    prediction_tuple = model_dni.predict(input_df_hora)[0] , model_ghi.predict(input_df_hora)[0]
                    prediction_hora = max(0, prediction_tuple[0] if target_radio == 'GHI' else prediction_tuple[1])
                    predictions_dia.append(prediction_hora)

            df_previsao_dia = pd.DataFrame(
                {f"{target_radio} Predito": predictions_dia}, 
                index=pd.to_datetime(timestamps_dia)
            )

            df_plot = df_previsao_dia
            
            # Junta com os dados reais (que já estão no session_state)
            if st.session_state.real_target_diaria is not None:
                real_column_name = f"{target_radio} Real"
                if real_column_name in st.session_state.real_target_diaria.columns:
                    df_plot = df_previsao_dia.join(st.session_state.real_target_diaria[real_column_name])
            
            # Salva o DataFrame do gráfico no session_state
            st.session_state.df_plot_diaria = df_plot

            
    # --- LÓGICA DE EXIBIÇÃO (SEMPRE ATIVA) ---
    # Esta seção está fora do "if st.button" e é desenhada em toda re-execução.
    with col_dia_2:
        st.subheader(f"Resultado da Simulação - {target_radio}")

        # 1. Desenha o gráfico de simulação se ele existir no session_state
        if not st.session_state.df_plot_diaria.empty:
            df_plot = st.session_state.df_plot_diaria
            y_label_text = f"{target_radio} (W/m²)"
            color_map = {'GHI': "#F87B1B", 'DNI': '#E6521F'}
            colors_to_use = []
            
            if f"{target_radio} Predito" in df_plot.columns:
                colors_to_use.append(color_map.get(target_radio, "#FB9E3A"))
            if f"{target_radio} Real" in df_plot.columns:
                colors_to_use.append("#FCEF91")

            if not colors_to_use:
                 colors_to_use = color_map.get(target_radio, "#EA2F14")
            
            st.line_chart(
                data=df_plot,
                color=colors_to_use,
                height=400,
                x_label="Hora",
                y_label=y_label_text
            )
        else:
            # Mensagem inicial antes do usuário clicar no botão
            st.info("Clique em 'Gerar Previsão Diária' para ver a simulação.")

        # 2. Desenha o Explorador de Features se os dados existirem no session_state
        if st.session_state.real_features_diaria is not None and not st.session_state.real_features_diaria.empty:
            st.markdown("---")
            st.subheader("Dados Médios Reais (2023)")
            st.markdown("Use o slider abaixo para explorar as condições meteorológicas reais médias registradas pelas estações durante o dia selecionado.")
            
            real_data_features_mean = st.session_state.real_features_diaria
            # Garante que o índice é um DatetimeIndex antes de acessar .hour
            dt_index = pd.DatetimeIndex(real_data_features_mean.index)
            daylight_hours = sorted({int(h) for h in dt_index.hour if 6 <= h <= 18})
            
            if daylight_hours:
                selected_hour = st.slider(
                    "Selecione uma hora:",
                    min_value=min(daylight_hours),
                    max_value=max(daylight_hours),
                    value=12 if 12 in daylight_hours else min(daylight_hours),
                    format="%02d:00",
                    key="explorer_slider"
                )
                try:
                    # Usa o DatetimeIndex para criar a máscara de hora
                    dt_index = pd.DatetimeIndex(real_data_features_mean.index)
                    mask = dt_index.hour == selected_hour
                    features_at_hour = real_data_features_mean[mask].iloc[0]
                    st.write(f"Condições médias reais às **{selected_hour:02d}:00**:")
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric(label="Temp. Ar (°C)", value=f"{features_at_hour.get('temp_ar', 'N/A'):.2f}")
                        st.metric(label="Vel. Vento (m/s)", value=f"{features_at_hour.get('vento_vel', 'N/A'):.2f}")
                    with c2:
                        st.metric(label="Umidade Rel. (%)", value=f"{features_at_hour.get('umidade_rel', 'N/A'):.2f}")
                        st.metric(label="Dir. Vento (°)", value=f"{features_at_hour.get('vento_dir', 'N/A'):.0f}")
                    with c3:
                        st.metric(label="Pressão (mB)", value=f"{features_at_hour.get('pressao_atm_estacao', 'N/A'):.2f}")
                        st.metric(label="Tipo de Nuvem", value=f"{features_at_hour.get('tipo_nuvem', 'N/A'):.0f}")
                    with c4:
                        st.metric(label="Precipitação (mm)", value=f"{features_at_hour.get('precipitacao', 'N/A'):.2f}")
                        
                
                except (IndexError, KeyError):
                    st.info("Não há dados de features diurnas para esta hora selecionada.")
            else:
                st.info("Não há dados de features diurnas para este dia.")
        
        # Se não for 2023, exibe uma mensagem
        elif data_simulacao.year != 2023:
             st.markdown("---")
             st.info("Selecione um dia em 2023 para comparar com os dados reais e explorar as features.")


# ==============================================================================
# ABA 3: ANÁLISE ANUAL
# ==============================================================================
with tab_anual:
    st.header("Análise de Desempenho e Simulação Anual")
    
    st.subheader("Comparativo no Ano de Validação (2023)")
    with st.spinner("Calculando previsões para 2023..."):
        if X_val is not None and y_val is not None:
            pred_val = pd.DataFrame({
                'ghi': model_ghi.predict(X_val[feature_names]),
                'dni': model_dni.predict(X_val[feature_names])
            }, index=y_val.index)
            df_monthly = pd.DataFrame({
                'GHI Real (Média)': y_val['ghi'].resample('ME').mean(),
                'GHI Predito (XGBoost)': pred_val['ghi'].resample('ME').mean()
            })
            st.line_chart(data=df_monthly, height=400, color=['#E6521F', '#FCEF91'], x_label="Data", y_label="GHI (W/m²)") # type: ignore
    
    st.markdown("---")
    st.subheader("Simulação para um Ano Futuro")
    st.markdown("Use a **barra lateral** para definir a localização e as condições de base. Opcionalmente, ajuste as médias climáticas anuais abaixo para criar cenários customizados.")
    
    future_year = st.number_input("Selecione o Ano para Simular", min_value=datetime.now().year, value=datetime.now().year + 1, step=1)
    
    with st.expander("Clique para ajustar as médias climáticas anuais"):
        # Define defaults a partir de X_val se disponível, caso contrário usa valores fallback seguros
        default_temp = round(X_val['temp_ar'].mean(), 1) if X_val is not None and 'temp_ar' in X_val.columns else 27.0
        default_humidity = round(X_val['umidade_rel'].mean(), 1) if X_val is not None and 'umidade_rel' in X_val.columns else 60.0

        # Sempre criar os sliders para garantir que user_temp e user_humidity sejam vinculados
        user_temp = st.slider("Temperatura Média Anual (°C)", 20.0, 35.0, default_temp)
        user_humidity = st.slider("Umidade Relativa Média Anual (%)", 40.0, 90.0, default_humidity)
        pressao_atm_media = st.slider("Pressão Atmosférica Média Estimada (mB)", 980.0, 1050.0, 1012.0)
        precipitacao_media = st.slider("Precipitação Média Anual (mm)", 0.0, 2000.0, 800.0)

    if st.button(f"Gerar Simulação para {future_year}", type="primary"):
        with st.spinner(f"Simulando todo o ano de {future_year}..."):
            future_dates = pd.date_range(start=f'{future_year}-01-01', end=f'{future_year}-12-31 23:00:00', freq='h')
            df_future = pd.DataFrame(index=future_dates)
            idx = pd.DatetimeIndex(df_future.index)
            df_future['hora_sin'] = np.sin(2 * np.pi * idx.hour / 24.0)
            df_future['hora_cos'] = np.cos(2 * np.pi * idx.hour / 24.0)
            df_future['dia_ano_sin'] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
            df_future['dia_ano_cos'] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
            df_future['latitude_inmet'] = lat
            df_future['longitude_inmet'] = lon
            df_future['temp_ar'] = user_temp
            df_future['umidade_rel'] = user_humidity
            df_future['vento_vel'] = user_wind_speed
            df_future['vento_dir'] = user_wind_dir
            df_future['tipo_nuvem'] = user_cloud_type
            df_future['pressao_atm_estacao'] = pressao_atm_media
            df_future['precipitacao'] = precipitacao_media / 8760.0  # Distribui a precipitação anual igualmente por hora
            
            df_future_final = df_future[feature_names]
            future_preds = pd.DataFrame({
                'ghi': model_ghi.predict(df_future_final),
                'dni': model_dni.predict(df_future_final)
            }, index=df_future_final.index)

            df_monthly_future = pd.DataFrame({'GHI Previsto (Simulação)': future_preds['ghi'].resample('M').mean()})
            st.write(f"**Média Mensal de GHI Previsto para {future_year} (W/m²)**")
            st.line_chart(data=df_monthly_future, height=400, use_container_width=True, color='#E6521F')

            df_kwh_future = pd.DataFrame({'Insolação Prevista (Simulação)': future_preds['ghi'].resample('M').sum() / 1000})
            st.write(f"**Insolação Total Mensal Prevista para {future_year} (kWh/m²)**")
            st.area_chart(data=df_kwh_future, height=400, use_container_width=True, color='#E6521F')

# ==============================================================================
# ABA 4: CHAT COM LLM
# ==============================================================================
with tab_chat:
    st.header("Chatbot do Projeto")
    st.markdown("Faça perguntas sobre o projeto, os modelos e os resultados")

    with st.form(key="chat_form", clear_on_submit=True):
            user_prompt = st.text_input("Faça sua pergunta:", placeholder="Ex: Qual modelo teve o menor erro?", key="chat_input")
            submit_button = st.form_submit_button("Enviar")

    for interaction in reversed(st.session_state.chat_history):
        with st.chat_message("user"):
            st.markdown(interaction["question"])
            
        with st.chat_message("model"):
            st.markdown(interaction["answer"])
            
            if interaction["images"]:
                for img_path in interaction["images"]:
                    if os.path.exists(img_path):
                        st.image(img_path, caption=f"Gráfico: {img_path}", width="content")
                    else:
                        st.warning(f"Imagem {img_path} não encontrada no servidor.")

    if submit_button and user_prompt:
        with st.spinner("Analisando sua pergunta e buscando gráficos..."):
            try:
                api_text, api_images = chat.run_ai(user_prompt)
                
                st.session_state.chat_history.append({
                    "question": user_prompt,
                    "answer": api_text,
                    "images": api_images
                })
                
            except Exception as e:
                error_message = f"Ocorreu um erro ao processar sua pergunta: {e}"
                st.session_state.chat_history.append({
                    "question": user_prompt,
                    "answer": error_message,
                    "images": []
                })
        
        st.rerun()

# ----------------------------------------------------------
# ABA 5 — SISTEMA FUZZY COMPLETO
# ----------------------------------------------------------
with aba_fuzzy:
    
    st.header("🔆 Sistema Fuzzy de Avaliação Solar")

    st.subheader("Entradas do Sistema Fuzzy")

    # ------------------------------------------------------
    #                     INPUTS DO USUÁRIO
    # ------------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        h_sin = st.slider("hora_sin", -1.0, 1.0, 0.0, 0.01)
        h_cos = st.slider("hora_cos", -1.0, 1.0, 0.0, 0.01)

    with col2:
        nuvem = st.slider("tipo_nuvem", 0.0, 10.0, 5.0, 0.1)
        temp = st.slider("temperatura do ar (°C)", 0.0, 40.0, 25.0, 0.1)

    # ------------------------------------------------------
    #               EXECUTA O MODELO FUZZY
    # ------------------------------------------------------
    resultado = avaliar_condicao_solar(h_sin, h_cos, nuvem, temp)

    st.subheader("Resultado da Defuzzificação")
    st.metric("Valor Crisp", f"{resultado:.2f}")

    st.subheader("Interpretação da Condição Solar")
    st.write(interpretar_condicao_fuzzy(resultado))

    st.markdown("---")

    # ------------------------------------------------------
    #         SEÇÃO DE GRÁFICOS (FUNÇÕES DE PERTINÊNCIA)
    # ------------------------------------------------------
    st.subheader("📈 Visualização das Funções de Pertinência")

    opcao = st.selectbox(
        "Escolha uma variável fuzzy para visualizar",
        ("hora_sin", "hora_cos", "tipo_nuvem", "temp_ar", "condicao_solar")
    )

    if opcao == "hora_sin":
        st.pyplot(plot_var(hora_sin))
    elif opcao == "hora_cos":
        st.pyplot(plot_var(hora_cos))
    elif opcao == "tipo_nuvem":
        st.pyplot(plot_var(tipo_nuvem))
    elif opcao == "temp_ar":
        st.pyplot(plot_var(temp_ar))
    elif opcao == "condicao_solar":
        st.pyplot(plot_var(condicao_solar))