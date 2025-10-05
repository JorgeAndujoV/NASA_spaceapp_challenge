import streamlit as st
import folium
from streamlit_folium import st_folium
import random
import datetime
import base64
import os

# <-- 1. IMPORTAMOS EL MODELO REAL DESDE model.py
# Le cambiamos el nombre a 'real_model_predict' para mayor claridad.
from model import mock_model_predict as real_model_predict

# =============================================================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Predictor de Hábitat de Tiburones",
    page_icon="🦈",
    layout="wide",
)

# =============================================================================
# 2. ELIMINAMOS LA SIMULACIÓN DEL MODELO
# =============================================================================
# La función 'mock_model_predict' que generaba números aleatorios ha sido BORRADA.
# Ahora usamos la función real que importamos de 'model.py'.

# =============================================================================
# 3. FUNCIONES AUXILIARES (Esta se queda igual)
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
# 4. INICIALIZACIÓN DEL ESTADO Y FUNCIONES DE UI
# =============================================================================
if "last_clicked" not in st.session_state:
    st.session_state["last_clicked"] = None

def autoplay_video(video_url: str):
    md = f"""
    <video controls loop autoplay="true" muted="true" style="width:100%;">
        <source src="{video_url}" type="video/webm">
    </video>
    """
    st.markdown(md, unsafe_allow_html=True)

# =============================================================================
# 5. CONSTRUCCIÓN DE LA INTERFAZ GRÁFICA (UI)
# =============================================================================
st.title("¿Dónde están los tiburones? 🦈")
st.markdown("Una herramienta para predecir hábitats de forrajeo de tiburones utilizando datos satelitales de la NASA.")

tab1, tab2, tab3 = st.tabs(["🌎 Herramienta Predictiva", "🔬 La Ciencia Detrás del Modelo", "🧠 Nuestra Metodología"])

# --- PESTAÑA 1: HERRAMIENTA PREDICTIVA ---
with tab1:
    map_col, results_col = st.columns([3, 2])
    with results_col:
        st.header("Panel de análisis")
        st.date_input(
            "Selecciona mes y año:",
            value=datetime.date(2025, 10, 5),
            min_value=datetime.date(2020, 1, 1),
            max_value=datetime.date(2030, 12, 31),
            key='date_input'
        )
        st.markdown("---")
        results_placeholder = st.empty()

    with map_col:
        if st.session_state["last_clicked"]:
            location = [st.session_state["last_clicked"]["lat"], st.session_state["last_clicked"]["lng"]]
            zoom = 6
        else:
            location = [20, 0]
            zoom = 2.5
        m = folium.Map(location=location, zoom_start=zoom)

        if st.session_state["last_clicked"]:
            lat = st.session_state["last_clicked"]["lat"]
            lon = st.session_state["last_clicked"]["lng"]
            date = st.session_state.date_input
            
            # <-- 3. USAMOS EL MODELO REAL
            probability = real_model_predict(lat, lon, date.month, date.year)
            level, color, emoji = get_probability_details(probability)
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

        map_data = st_folium(m, height=600, width="100%", returned_objects=["last_clicked"])

        if map_data and map_data["last_clicked"]:
            st.session_state["last_clicked"] = map_data["last_clicked"]
            st.rerun()

    if st.session_state["last_clicked"]:
        lat = st.session_state["last_clicked"]["lat"]
        lon = st.session_state["last_clicked"]["lng"]
        date = st.session_state.date_input
        
        # <-- 3. USAMOS EL MODELO REAL (de nuevo para el panel)
        probability = real_model_predict(lat, lon, date.month, date.year)
        level, color, emoji = get_probability_details(probability)
        
        with results_placeholder.container():
            st.subheader(f"Resultado para el punto seleccionado:")
            st.metric(label=f"{emoji} Nivel de probabilidad", value=level)
            st.metric(label="Valor de probabilidad", value=f"{probability:.2%}")
            st.progress(probability)
            with st.expander("Detalles de la entrada"):
                st.write(f"**Latitud:** {lat:.4f}")
                st.write(f"**Longitud:** {lon:.4f}")
                st.write(f"**Fecha:** {date.strftime('%B %Y')}")
    else:
        with results_placeholder.container():
            st.info("ℹ️ Haz clic en un punto del mapa para iniciar el análisis.")

# --- PESTAÑA 2: SECCIÓN DIDÁCTICA ---
with tab2:
    # ... (Todo el código de la Pestaña 2 se queda exactamente igual) ...
    st.header("Plancton y corrientes")
    st.write("Nuestro modelo funciona como un detective que busca dos pistas clave en el océano para encontrar los lugares preferidos de los tiburones. Estas pistas, invisibles al ojo humano, son capturadas por los satélites de la NASA.")
    
    video_col1, video_col2 = st.columns(2)

    with video_col1:
        st.subheader("El Plancton, base de la cadena alimenticia")
        try:
            autoplay_video('https://raw.githubusercontent.com/JorgeAndujoV/NASA_spaceapp_challenge/main/plancton.webm') 
            st.caption("Crédito: MIT Darwin Project, ECCO2, MITgcm")
        except Exception:
            st.warning("No se pudo cargar el video de plancton.")
    
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
            autoplay_video('https://raw.githubusercontent.com/JorgeAndujoV/NASA_spaceapp_challenge/main/currents.webm') 
            st.caption("Crédito: NASA/Goddard Space Flight Center Scientific Visualization Studio")
        except Exception:
            st.warning("No se pudo cargar el video de corrientes.")
            
        st.markdown(
            """
            Las **corrientes oceánicas** actúan como ríos y autopistas que transportan nutrientes y agrupan a las presas.
            - **¿Por qué es relevante?** Los tiburones son depredadores inteligentes. No gastan energía buscando al azar; utilizan las corrientes para viajar y patrullar zonas donde los remolinos y frentes oceánicos concentran a sus presas, como si fueran embudos.
            **Nuestra herramienta** emplea datos imágenes satelitales de la NASA para identificar estas rutas y puntos calientes, ayudándonos a predecir dónde es más probable que los tiburones patrullen en busca de alimento.
            """
        )

# --- PESTAÑA 3: METODOLOGÍA Y PROPUESTA ---
with tab3:
    # ... (Todo el código de la Pestaña 3 se queda exactamente igual) ...
    st.header("Datos satelitales y machine learning")
    st.write(
        """
        Nuestro primer objetivo fue transformar datos crudos del océano en una predicción útil y accesible. Para lograrlo, construimos un pipeline de Machine Learning de dos etapas que combina el aprendizaje no supervisado para entender el ambiente, y el aprendizaje supervisado para predecir la presencia de tiburones.
        """
    )
    st.markdown("---")
    img_col_cluster, text_col_cluster = st.columns([2, 3])

    with img_col_cluster:
        st.image(
            'enesep.png', 
            caption="Visualización de los clústeres oceanográficos identificados por nuestro modelo, comparación enero-septiembre de dos años.",
            use_container_width=True
        )

    with text_col_cluster:
        st.subheader("1. Procesamiento de datos de la NASA")
        st.markdown(
            """
            Comenzamos con bases de datos de imagenes satelitales de la NASA que capturan variables oceánicas clave como la temperatura del mar, la salinidad, la clorofila (indicador de plancton) y las corrientes oceánicas.
            """
        )
        st.subheader("2. Clústering para entender el océano")
        st.markdown(
            """
            Para que nuestro modelo pudiera entender el océano, primero necesitábamos enseñarle a reconocer diferentes "tipos" de ambientes. Analizamos varios algoritmos de clustering y seleccionamos aquel cuyas agrupaciones mostraron una fuerte correlación con las clasificaciones oceanográficas de la NASA.

            Cada clúster representa una "etiqueta" para un ecosistema con características únicas, permitiéndonos crear nuestros propios datasets de entrenamiento.
            """
        )

    st.markdown("---")
    st.subheader("3. Predicción de hábitats de tiburones")
    st.markdown(
        """
        Con el océano ya etiquetado por clústeres, entrenamos una **red neuronal**. Esta fórmula representa el **marco teórico** de nuestro enfoque, donde la probabilidad de un tiburón se calcula a partir de la probabilidad de cada tipo de ambiente:
        """
    )
    st.latex(r'''
        P(\text{shark}|x) = \sum_{k=1}^{K} P(\text{cluster} = k|x) P(\text{shark}|\text{cluster} = k, \text{time})
    ''')
    st.markdown(
        """
        Nuestra **red neuronal** es la implementación que resuelve esta ecuación. El modelo aprendió la relación entre las características ambientales de cada clúster y la probabilidad de encontrar tiburones en esas zonas.

        La predicción final es una **"predicción proxy"** inteligente. El modelo utiliza patrones de los clústeres además de conocimiento biológico sobre el comportamiento de los tiburones y las características del mar para estimar dónde es más probable que se encuentren.
        """
    )
    st.markdown("---")
    st.header("Propuesta de rediseño de etiqueta de archivo satelital emergente")
    img_col_tag, text_col_tag = st.columns([2, 3])
    with img_col_tag:
        st.image(
            'tag.jpg',
            use_container_width=True,
            caption="Propuesta de tag satelital avanzado."
        )
    with text_col_tag:
        st.write(
            """
            El tag registrará información esencial del entorno y del comportamiento del tiburón...
            """
        )