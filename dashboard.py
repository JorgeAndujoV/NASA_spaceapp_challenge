# app.py

import streamlit as st
import folium
from streamlit_folium import st_folium
import random
import datetime

# =============================================================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Predictor de Hábitat de Tiburones",
    page_icon="🦈",
    layout="wide",
)

# =============================================================================
# 2. SIMULACIÓN DEL MODELO DE PREDICCIÓN (MOCK MODEL)
# =============================================================================
# ESTA ES LA FUNCIÓN QUE DEBERÁS REEMPLAZAR CON TU MODELO REAL
def mock_model_predict(lat: float, lon: float, month: int, year: int) -> float:
    """
    Simula la predicción de un modelo de ML.
    Toma coordenadas y fecha y devuelve una probabilidad entre 0.0 y 1.0.
    Usamos la entrada como 'seed' para que el resultado sea aleatorio pero consistente.
    """
    # La semilla asegura que para la misma lat/lon/fecha, siempre obtengas el mismo número "aleatorio"
    seed = f"{lat}{lon}{month}{year}"
    random.seed(seed)
    # Genera la probabilidad
    probability = random.uniform(0.0, 1.0)
    return probability

# =============================================================================
# 3. FUNCIONES AUXILIARES
# =============================================================================
def get_probability_details(probability: float) -> tuple:
    """
    Categoriza la probabilidad en niveles y asigna un color y emoji.
    """
    if probability < 0.33:
        level = "Baja"
        color = "green"
        emoji = "🟢"
    elif probability < 0.66:
        level = "Media"
        color = "orange"
        emoji = "🟠"
    else:
        level = "Alta"
        color = "red"
        emoji = "🔴"
    return level, color, emoji

# =============================================================================
# 4. INICIALIZACIÓN DEL ESTADO DE LA APLICACIÓN
# =============================================================================
# 'st.session_state' se usa para guardar información entre interacciones del usuario.
if "last_clicked" not in st.session_state:
    st.session_state["last_clicked"] = None

# =============================================================================
# 5. CONSTRUCCIÓN DE LA INTERFAZ GRÁFICA (UI)
# =============================================================================

# --- TÍTULO ---
st.title("¿Dónde están los tiburones? 🦈")
st.markdown("Haz clic en cualquier punto del mapa y selecciona una fecha para analizar la probabilidad de forrajeo.")

# --- LAYOUT DE DOS COLUMNAS ---
map_col, results_col = st.columns([3, 2]) # El mapa ocupa más espacio

# --- COLUMNA 1: MAPA INTERACTIVO ---
with map_col:
    # Crear un mapa de Folium centrado en una vista global
    m = folium.Map(location=[20, 0], zoom_start=2.5)

    # Si hay un punto seleccionado, procesarlo y mostrarlo
    if st.session_state["last_clicked"]:
        lat = st.session_state["last_clicked"]["lat"]
        lon = st.session_state["last_clicked"]["lng"]
        
        # Obtener la fecha del selector en la otra columna
        date_input = st.session_state.get('date_input', datetime.date.today())
        month = date_input.month
        year = date_input.year
        
        # Llamar al modelo para obtener la predicción
        probability = mock_model_predict(lat, lon, month, year)
        level, color, emoji = get_probability_details(probability)
        
        # Crear un marcador en el mapa
        popup_text = f"""
        <b>Ubicación Analizada</b><br>
        Lat: {lat:.2f}, Lon: {lon:.2f}<br>
        Probabilidad: <b>{probability:.0%} ({level})</b>
        """
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_text, max_width=300),
            icon=folium.Icon(color=color, icon="life-ring", prefix='fa')
        ).add_to(m)

    # Renderizar el mapa y capturar el último clic
    map_data = st_folium(m, height=600, width="100%", returned_objects=["last_clicked"])

    # Actualizar la ubicación del clic en el estado de la sesión
    if map_data and map_data["last_clicked"]:
        st.session_state["last_clicked"] = map_data["last_clicked"]


# --- COLUMNA 2: CONTROLES Y RESULTADOS ---
with results_col:
    st.header("Panel de Análisis")

    # Selector de fecha
    st.date_input(
        "Selecciona Mes y Año:",
        value=datetime.date.today(),
        min_value=datetime.date(2020, 1, 1),
        max_value=datetime.date(2030, 12, 31),
        key='date_input' # Usamos una 'key' para acceder al valor desde otras partes del código
    )

    st.markdown("---")

    # Mostrar los resultados si hay un punto seleccionado
    if st.session_state["last_clicked"]:
        lat = st.session_state["last_clicked"]["lat"]
        lon = st.session_state["last_clicked"]["lng"]
        date = st.session_state.date_input
        
        # Recalcular la probabilidad para mostrarla en el panel
        probability = mock_model_predict(lat, lon, date.month, date.year)
        level, color, emoji = get_probability_details(probability)
        
        st.subheader(f"Resultado para el Punto Seleccionado:")
        st.metric(label=f"{emoji} Nivel de Probabilidad", value=level)
        st.metric(label="Valor de Probabilidad", value=f"{probability:.2%}")
        st.progress(probability)
        
        with st.expander("Detalles de la Entrada"):
            st.write(f"**Latitud:** {lat:.4f}")
            st.write(f"**Longitud:** {lon:.4f}")
            st.write(f"**Fecha:** {date.strftime('%B %Y')}")
    else:
        st.info("ℹ️ Haz clic en un punto del mapa para iniciar el análisis.")