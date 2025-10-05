import streamlit as st
import folium
from streamlit_folium import st_folium
import random
import datetime
import base64
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

# --- FUNCIÓN AUXILIAR PARA VIDEOS CON AUTOPLAY Y LOOP ---
# La he movido aquí para que esté disponible para la pestaña 2
def autoplay_video(video_url: str):
    # Simplemente genera el HTML con la URL del video
    md = f"""
    <video controls loop autoplay="true" muted="true" style="width:100%;">
        <source src="{video_url}" type="video/webm">
    </video>
    """
    st.markdown(md, unsafe_allow_html=True)

# =============================================================================
# 5. CONSTRUCCIÓN DE LA INTERFAZ GRÁFICA (UI)
# =============================================================================

# --- TÍTULO ---
st.title("¿Dónde están los tiburones? 🦈")
st.markdown("Una herramienta para predecir hábitats de forrajeo de tiburones utilizando datos satelitales de la NASA.")

# --- ESTRUCTURA DE PESTAÑAS PARA ORGANIZAR EL CONTENIDO ---
tab1, tab2 = st.tabs(["🌎 Herramienta predictiva", "🔬 La ciencia detrás del modelo"])

# --- PESTAÑA 1: HERRAMIENTA PREDICTIVA (Tu código original - NO HA CAMBIADO) ---
with tab1:
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
        st.header("Panel de análisis")

        # Selector de fecha
        st.date_input(
            "Selecciona mes y año:",
            value=datetime.date(2025, 10, 5), # Fecha actual
            min_value=datetime.date(2020, 1, 1),
            max_value=datetime.date(2030, 12, 31),
            key='date_input'
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
            
            st.subheader(f"Resultado para el punto seleccionado:")
            st.metric(label=f"{emoji} Nivel de probabilidad", value=level)
            st.metric(label="Valor de probabilidad", value=f"{probability:.2%}")
            st.progress(probability)
            
            with st.expander("Detalles de la entrada"):
                st.write(f"**Latitud:** {lat:.4f}")
                st.write(f"**Longitud:** {lon:.4f}")
                st.write(f"**Fecha:** {date.strftime('%B %Y')}")
        else:
            st.info("ℹ️ Haz clic en un punto del mapa para iniciar el análisis.")

# --- PESTAÑA 2: SECCIÓN DIDÁCTICA (AHORA CON AUTOPLAY Y TU TEXTO ACTUALIZADO) ---
with tab2:
    st.header("Plancton y corrientes")
    st.write("Nuestro modelo funciona como un detective que busca dos pistas clave en el océano para encontrar los lugares preferidos de los tiburones. Estas pistas, invisibles al ojo humano, son capturadas por los satélites de la NASA.")
    
    # --- Layout de dos columnas para los videos ---
    video_col1, video_col2 = st.columns(2)

    with video_col1:
        st.subheader("El Plancton, base de la cadena alimenticia")
        try:
            # USANDO LA FUNCIÓN AUTOPLAY_VIDEO
            autoplay_video('https://raw.githubusercontent.com/JorgeAndujoV/NASA_spaceapp_challenge/main/plancton.webm') 
            st.caption("Crédito: MIT Darwin Project, ECCO2, MITgcm")
        except FileNotFoundError:
            st.error("No se encontró el archivo 'plancton.webm'. Asegúrate de que esté en la misma carpeta que tu script.")
            
        st.markdown(
            """
            El **fitoplancton** son algas microscópicas que forman la base de toda la cadena alimenticia en el océano. 
            - **¿Por qué es relevante?** Donde hay grandes concentraciones de plancton, hay pequeños peces y crustáceos alimentándose. Esto atrae a peces más grandes, que a su vez, son el alimento principal de los tiburones.
            **Nuestra herramienta** utiliza las imágenes satelitales de clorofila para identificar estas áreas ricas en plancton, ayudándonos a predecir dónde es más probable que los tiburones encuentren comida.
            """
        )

    with video_col2:
        st.subheader("Las corrientes, autopistas del océano")
        try:
            # USANDO LA FUNCIÓN AUTOPLAY_VIDEO
            autoplay_video('https://raw.githubusercontent.com/JorgeAndujoV/NASA_spaceapp_challenge/main/currents.webm') 
            st.caption("Crédito: NASA/Goddard Space Flight Center Scientific Visualization Studio")
        except FileNotFoundError:
            st.error("No se encontró el archivo 'currents.webm'. Asegúrate de que esté en la misma carpeta que tu script.")
            
        st.markdown(
            """
            Las **corrientes oceánicas** actúan como ríos y autopistas que transportan nutrientes y agrupan a las presas.
            - **¿Por qué es relevante?** Los tiburones son depredadores inteligentes. No gastan energía buscando al azar; utilizan las corrientes para viajar y patrullar zonas donde los remolinos y frentes oceánicos concentran a sus presas, como si fueran embudos.
            **Nuestra herramienta** emplea datos imágenes satelitales de la NASA para identificar estas rutas y puntos calientes, ayudándonos a predecir dónde es más probable que los tiburones patrullen en busca de alimento.
            """
        )
    
    st.markdown("---")
    
    st.header("¿Por qué es importante encontrar a los tiburones?")
    st.write(
        """
        Encontrar a los tiburones no es solo para satisfacer nuestra curiosidad. Los tiburones son **depredadores tope** y, como tales, son **bioindicadores cruciales de la salud del océano**.
        """
    )
    
    st.subheader("Indicadores de ecosistemas saludables")
    st.markdown(
        """
        Una población saludable de tiburones en una zona indica que toda la cadena alimenticia debajo de ellos es robusta y está en equilibrio. Si los tiburones desaparecen de un área, puede ser una señal de alerta temprana de problemas más graves como la sobrepesca, la contaminación o los efectos del cambio climático en los niveles más bajos del ecosistema.
        """
    )
    
    st.subheader("¿Qué problema estamos resolviendo?")
    st.markdown(
        """
        - **Conservación Eficiente:** Al predecir sus hábitats de alimentación, podemos ayudar a diseñar **Áreas Marinas Protegidas (AMP)** que sean dinámicas y más efectivas, protegiendo los lugares donde los tiburones son más vulnerables.
        - **Monitoreo Climático:** Rastrear cómo cambian estos hábitats a lo largo del tiempo nos proporciona datos valiosos sobre cómo el cambio climático está impactando la vida marina y la distribución de especies.
        - **Mitigación de la Pesca Incidental:** Nuestra herramienta podría alertar a las flotas pesqueras sobre zonas de alta probabilidad de actividad de tiburones, ayudando a reducir el número de tiburones capturados accidentalmente.
        """
    )
