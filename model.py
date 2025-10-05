import numpy as np
import joblib
import json
import os
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import hashlib

# Configuración global del modelo
_MODEL_LOADED = False
_MODEL = None
_SCALER = None
_W_K = None

def _load_model_once():
    """
    Carga el modelo una sola vez para optimizar rendimiento
    """
    global _MODEL_LOADED, _MODEL, _SCALER, _W_K
    
    if _MODEL_LOADED:
        return
    
    try:
        # Ruta específica de tu modelo
        MODEL_DIR = 'shark_model_optimized_v3.0.0_20251005_192236'
        
        print("Cargando modelo de predicción de tiburones...")
        
        # 1. Cargar modelo Keras
        model_path = os.path.join(MODEL_DIR, "shark_prediction_model.h5")
        _MODEL = load_model(model_path)
        print(f"Modelo cargado: {os.path.basename(model_path)}")
        
        # 2. Cargar scaler
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        _SCALER = joblib.load(scaler_path)
        print("Scaler cargado")
        
        # 3. Cargar pesos de clusters
        weights_path = os.path.join(MODEL_DIR, "cluster_weights.json")
        with open(weights_path, 'r') as f:
            _W_K = json.load(f)
        print("Pesos de clusters cargados")
        
        _MODEL_LOADED = True
        print("Modelo listo para predicciones")
        
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        print("🔄 Usando modo fallback para predicciones")
        _MODEL_LOADED = False

def _prepare_features(lat: float, lon: float, month: int, year: int):
    """
    Prepara las características en el mismo formato que durante el entrenamiento
    """
    # Características temporales
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    # Factor latitudinal
    abs_lat = np.abs(lat)
    lat_factor = np.exp(-0.5 * ((abs_lat - 50) / 20) ** 2)
    latitudinal_factor = 0.5 + 0.5 * lat_factor
    
    # Intensidad estacional
    high_season_months = [11, 12, 1, 2, 3]
    seasonal_intensity = 1.0 if month in high_season_months else 0.3
    
    # Interacciones
    lat_month_interaction = lat * month_sin
    lon_month_interaction = lon * month_cos
    
    # Características adicionales
    distance_equator = np.abs(lat)
    ocean_region = 0 if lon < 0 else 1  # 0: Pacífico Este, 1: Oeste
    
    # Cluster placeholder (será ignorado en predicción)
    cluster = 0
    
    # Combinar características (mismo orden que durante entrenamiento)
    features = np.array([[
        lat, lon, month, year,
        month_sin, month_cos,
        latitudinal_factor,
        cluster,
        lat_month_interaction,
        lon_month_interaction,
        seasonal_intensity,
        distance_equator,
        ocean_region
    ]], dtype=np.float32)
    
    return features

def _fallback_prediction(lat: float, lon: float, month: int, year: int) -> float:
    """
    Predicción de fallback consistente (similar al mock original)
    """
    seed_str = f"{lat:.6f}{lon:.6f}{month}{year}"
    seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
    
    np.random.seed(seed)
    return float(np.random.uniform(0.0, 1.0))

def _categorize_risk(risk_score: float) -> str:
    """Categoriza el riesgo en niveles"""
    if risk_score >= 0.7: return "MUY ALTO"
    elif risk_score >= 0.5: return "ALTO"
    elif risk_score >= 0.3: return "MEDIO"
    elif risk_score >= 0.1: return "BAJO"
    else: return "MUY BAJO"

def _categorize_confidence(confidence: float) -> str:
    """Categoriza la confianza de la predicción"""
    if confidence >= 0.9: return "MUY ALTA"
    elif confidence >= 0.7: return "ALTA"
    elif confidence >= 0.5: return "MEDIA"
    else: return "BAJA"

def mock_model_predict(lat: float, lon: float, month: int, year: int) -> float:
    """
    Predice el riesgo de tiburones usando el modelo entrenado.
    Toma coordenadas y fecha y devuelve una probabilidad entre 0.0 y 1.0.
    
    Args:
        lat: Latitud (-90 a 90)
        lon: Longitud (-180 a 180)
        month: Mes (1-12)
        year: Año
    
    Returns:
        float: Probabilidad de riesgo entre 0.0 y 1.0
    """
    # Cargar modelo si no está cargado
    if not _MODEL_LOADED:
        _load_model_once()
    
    # Si el modelo no se pudo carrar, usar fallback
    if not _MODEL_LOADED:
        return _fallback_prediction(lat, lon, month, year)
    
    try:
        # Preparar características
        features = _prepare_features(lat, lon, month, year)
        
        # Estandarizar características
        features_scaled = _SCALER.transform(features).astype(np.float32)
        
        # Hacer predicción
        cluster_probs = _MODEL.predict(features_scaled, verbose=0)
        
        # Calcular riesgo de tiburones usando pesos de clusters
        if _W_K:
            w_vector = np.array([_W_K[str(i)] for i in range(len(_W_K))])
            shark_risk = np.dot(cluster_probs, w_vector)[0]
        else:
            # Si no hay pesos, usar probabilidad promedio
            shark_risk = np.mean(cluster_probs)
        
        # Asegurar que el riesgo esté entre 0 y 1
        shark_risk = float(np.clip(shark_risk, 0.0, 1.0))
        
        return shark_risk
        
    except Exception as e:
        print(f"Error en predicción: {e}, usando fallback")
        return _fallback_prediction(lat, lon, month, year)

# VERSIÓN MEJORADA CON MÁS INFORMACIÓN (opcional)
def enhanced_shark_predict(lat: float, lon: float, month: int, year: int) -> dict:
    """
    Versión mejorada que retorna más información sobre la predicción
    
    Returns:
        dict: Diccionario con información detallada de la predicción
    """
    # Cargar modelo si no está cargado
    if not _MODEL_LOADED:
        _load_model_once()
    
    # Predicción base
    risk_score = mock_model_predict(lat, lon, month, year)
    
    # Información adicional
    result = {
        'risk_probability': risk_score,
        'risk_category': _categorize_risk(risk_score),
        'coordinates': {'lat': lat, 'lon': lon},
        'date': {'month': month, 'year': year},
        'model_used': _MODEL_LOADED,
        'confidence': 'ALTA' if _MODEL_LOADED else 'BAJA',
        'seasonal_factor': 'ALTA' if month in [11, 12, 1, 2, 3] else 'BAJA',
        'latitude_zone': 'TROPICAL' if abs(lat) <= 23.5 else 'TEMPERATE' if abs(lat) <= 66.5 else 'POLAR'
    }
    
    # Si el modelo está cargado, añadir información de clusters
    if _MODEL_LOADED:
        try:
            features = _prepare_features(lat, lon, month, year)
            features_scaled = _SCALER.transform(features).astype(np.float32)
            cluster_probs = _MODEL.predict(features_scaled, verbose=0)[0]
            
            result.update({
                'predicted_cluster': int(np.argmax(cluster_probs)),
                'cluster_confidence': float(np.max(cluster_probs)),
                'cluster_probabilities': cluster_probs.tolist(),
                'top_clusters': [
                    {'cluster': i, 'probability': float(prob)} 
                    for i, prob in sorted(enumerate(cluster_probs), 
                                        key=lambda x: x[1], reverse=True)[:3]
                ]
            })
        except Exception as e:
            result['cluster_info'] = f"Error: {e}"
    
    return result

# EJEMPLOS DE USO
if __name__ == "__main__":
    # Ejemplo de uso básico (igual que tu función original)
    print("PREDICCIONES BÁSICAS:")
    test_cases = [
        (-17.54167, -107.70833, 1, 2024),
        (-18.04167, 163.45834, 1, 2024),
        (34.0, -118.0, 6, 2024)
    ]
    
    for lat, lon, month, year in test_cases:
        risk = mock_model_predict(lat, lon, month, year)
        print(f" ({lat:.2f}, {lon:.2f}) |  {month}/{year} |  Riesgo: {risk:.4f}")
    
    print("\nPREDICCIONES DETALLADAS:")
    # Ejemplo de uso mejorado
    detailed = enhanced_shark_predict(-17.54167, -107.70833, 1, 2024)
    print("Predicción detallada:")
    for key, value in detailed.items():
        print(f"  {key}: {value}")