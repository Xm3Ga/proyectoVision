"""
Configuración del sistema de reconocimiento de cartas.
Ajustar estos valores según las condiciones de iluminación y cámara.
"""

# =============================================================================
# CONFIGURACIÓN DE CÁMARA
# =============================================================================
CAMERA_INDEX = 1  # Índice de la cámara (1 para Camo)
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# =============================================================================
# CONFIGURACIÓN DE DETECCIÓN DEL TAPETE VERDE
# =============================================================================
# Rango HSV para detectar el color verde del tapete
# H: 0-180, S: 0-255, V: 0-255 en OpenCV
GREEN_HSV_LOWER = (35, 40, 40)   # Límite inferior HSV para verde
GREEN_HSV_UPPER = (85, 255, 255)  # Límite superior HSV para verde

# =============================================================================
# CONFIGURACIÓN DE DETECCIÓN DE CARTAS
# =============================================================================
# Área mínima y máxima de contornos para considerar como carta (en píxeles)
CARD_MIN_AREA = 5000
CARD_MAX_AREA = 500000

# Relación de aspecto de una carta estándar (ancho/alto)
# Carta estándar: 63mm x 88mm = 0.716
CARD_ASPECT_RATIO = 0.716
CARD_ASPECT_TOLERANCE = 0.3  # Tolerancia para variaciones

# Tamaño normalizado de carta para procesamiento (ancho x alto)
CARD_WIDTH = 200
CARD_HEIGHT = 280

# =============================================================================
# CONFIGURACIÓN DE RECONOCIMIENTO
# =============================================================================
# Región de interés para el valor/palo (esquina superior izquierda)
# Como porcentaje del tamaño de la carta normalizada
ROI_VALUE_X = 0.02
ROI_VALUE_Y = 0.02
ROI_VALUE_WIDTH = 0.22
ROI_VALUE_HEIGHT = 0.35

# Umbral para template matching (0-1, mayor = más estricto)
MATCH_THRESHOLD_VALUE = 0.20
MATCH_THRESHOLD_SUIT = 0.15

# =============================================================================
# NOMBRES DE VALORES Y PALOS
# =============================================================================
CARD_VALUES = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
CARD_SUITS = ['corazones', 'diamantes', 'treboles', 'picas']
SUIT_SYMBOLS = {'corazones': '♥', 'diamantes': '♦', 'treboles': '♣', 'picas': '♠'}

# =============================================================================
# VISUALIZACIÓN
# =============================================================================
SHOW_DEBUG_WINDOWS = True  # Mostrar ventanas de depuración
FONT_SCALE = 0.8
FONT_THICKNESS = 2

