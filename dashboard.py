# app.py

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import time
import datetime

# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="Apex-AI: Predictor Global de Hábitats de Tiburones",
    page_icon="🦈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Data Loading (Use caching to speed up app)
# =============================================================================
# En un app real, cargarías los datos de predicción de tu modelo aquí.
# Para este ejemplo, generaremos algunos datos falsos.
@st.cache_data
def load_data():
    # Fake historical sightings data (for optional display)
    sightings = pd.DataFrame({
        'lat': np.random.uniform(-60, 60, 100), # Global distribution for world map
        'lon': np.random.uniform(-180, 180, 100),
        'species': np.random.choice(['Great White Shark', 'Tiger Shark', 'Whale Shark'], 100)
    })
    return sightings

sightings_df = load_data()

# =============================================================================
# Título y Descripción General
# =============================================================================
st.title("🦈 Apex-AI: Predicción Global de Hábitats de Tiburones")
st.markdown("Explora el mapa mundial, haz clic en cualquier punto del océano y selecciona una fecha para predecir la probabilidad de actividad de forrajeo de tiburones.")

# =============================================================================
# Diseño en Columnas
# =============================================================================
col1, col2 = st.columns([3, 2]) # Columna del mapa más ancha que la de controles/resultados

# --- Columna 1: Mapa Interactivo Mundial ---
with col1:
    st.subheader("Mapa Global Interactivo")

    # Crear un mapa de Folium centrado en una vista global
    m = folium.Map(location=[0, 0], zoom_start=2, control_scale=True)

    # Opcional: Añadir capas de avistamientos históricos
    if st.session_state.get('show_sightings', False):
        for idx, row in sightings_df.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=3,
                color='red',
                fill=True,
                fill_color='red',
                tooltip=f"Avistamiento: {row['species']}"
            ).add_to(m)

    # Renderizar el mapa de Folium en Streamlit y capturar clics
    # La clave aquí es 'return_on_hover=True' para la interactividad de coordenadas
    # Aunque la captura de clic directo se maneja con 'st_folium'
    map_data = st_folium(
        m,
        height=600,
        width="100%",
        feature_group_to_add=None,
        returned_objects=["last_clicked"], # Devuelve la última ubicación clicada
        key="global_map" # Clave única para el componente Streamlit
    )

# --- Columna 2: Controles y Resultados de Predicción ---
with col2:
    st.subheader("Análisis de Punto y Fecha")

    # Selector de fecha
    selected_date = st.date_input("Selecciona una fecha para el análisis:", datetime.date.today())

    st.markdown("---")
    st.markdown("### Coordenadas del Punto Seleccionado:")

    # Mostrar las coordenadas del último clic
    clicked_lat = None
    clicked_lon = None

    if map_data and map_data["last_clicked"]:
        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lon = map_data["last_clicked"]["lng"]
        st.write(f"**Latitud:** {clicked_lat:.4f}")
        st.write(f"**Longitud:** {clicked_lon:.4f}")
    else:
        st.write("Haz clic en un punto del mapa para seleccionarlo.")

    # Botón de análisis (solo si hay un punto seleccionado)
    if clicked_lat is not None and clicked_lon is not None:
        if st.button("✨ Obtener Predicción"):
            # --- Lógica de Predicción ---
            # Aquí es donde integrarías tu modelo de TensorFlow
            # 1. Obtener los datos ambientales (PACE, SWOT, etc.) para (clicked_lat, clicked_lon, selected_date)
            #    Ejemplo: `environmental_features = get_environmental_data_for_point(clicked_lat, clicked_lon, selected_date)`

            # 2. Usar tu modelo de TensorFlow para predecir la probabilidad
            #    Ejemplo: `prediction_probability = your_tensorflow_model.predict(environmental_features)`

            # --- SIMULACIÓN DE PREDICCIÓN (REEMPLAZA ESTO CON TU CÓDIGO REAL) ---
            # Usamos un poco de aleatoriedad para simular diferentes probabilidades
            np.random.seed(int(clicked_lat * 1000) + int(clicked_lon * 1000) + selected_date.day)
            prediction_probability = np.random.uniform(0.1, 0.99) # Probabilidad aleatoria para demostración
            # --- FIN SIMULACIÓN ---

            st.write("---")
            st.write("### Resultado de la Predicción:")

            if prediction_probability > 0.6: # Umbral de ejemplo para "POSIBLE"
                st.success(f"**¡HÁBITAT POSIBLE para forrajeo de tiburones!**")
                st.metric(label="Probabilidad de Actividad", value=f"{prediction_probability:.0%}")
                st.info("Condiciones favorables esperadas: Alta concentración de fitoplancton, corrientes oceánicas convergentes.") # Aquí añadirías los fenómenos climáticos relevantes
            else:
                st.warning(f"**HÁBITAT POCO PROBABLE**")
                st.metric(label="Probabilidad de Actividad", value=f"{prediction_probability:.0%}")
                st.info("Condiciones desfavorables esperadas: Baja productividad primaria, corrientes dispersas.") # Aquí añadirías los fenómenos climáticos relevantes
        else:
            st.info("Haz clic en 'Obtener Predicción' para ver el análisis.")
    else:
        st.info("Primero haz clic en un punto del mapa para seleccionarlo y luego podrás obtener la predicción.")

    st.sidebar.markdown("---")
    st.sidebar.header("Opciones de Visualización")
    st.session_state['show_sightings'] = st.sidebar.checkbox("Mostrar Avistamientos Históricos (OBIS)", value=True)
    # Aquí puedes añadir más checkboxes para capas de datos de PACE/SWOT si quieres visualizarlas globalmente