"""
Módulo de detección de cartas en la imagen.
Utiliza segmentación por color HSV y análisis de contornos.
"""

import cv2
import numpy as np
from config import (
    GREEN_HSV_LOWER, GREEN_HSV_UPPER,
    CARD_MIN_AREA, CARD_MAX_AREA,
    CARD_ASPECT_RATIO, CARD_ASPECT_TOLERANCE,
    CARD_WIDTH, CARD_HEIGHT
)


def preprocess_image(image):
    """
    Preprocesa la imagen para mejorar la detección.
    """
    # Aplicar desenfoque gaussiano para reducir ruido
    # Kernel 5x5: tamaño moderado que suaviza sin perder detalles importantes
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Convertir a HSV para segmentación por color
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    return blurred, hsv


def segment_green_background(hsv_image):
    """
    Segmenta el fondo verde del tapete.    
    """
    # Crear máscara para el color verde
    green_mask = cv2.inRange(hsv_image, GREEN_HSV_LOWER, GREEN_HSV_UPPER)
    
    # Invertir la máscara: queremos que las cartas (no verdes) sean blancas
    card_mask = cv2.bitwise_not(green_mask)
    
    # Operaciones morfológicas para limpiar la máscara
    # Kernel elíptico: mejor para objetos con bordes suaves
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Apertura: elimina pequeños puntos blancos (ruido)
    card_mask = cv2.morphologyEx(card_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Cierre: rellena pequeños agujeros negros dentro de las cartas
    card_mask = cv2.morphologyEx(card_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return card_mask


def find_card_contours(mask):
    """
    Encuentra contornos que podrían ser cartas.
    """
    # Encontrar contornos externos
    # RETR_EXTERNAL: solo contornos externos (ignora agujeros)
    # CHAIN_APPROX_SIMPLE: comprime segmentos horizontales/verticales
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filtrar por área
        if CARD_MIN_AREA < area < CARD_MAX_AREA:
            valid_contours.append(contour)
    
    # Ordenar por área (mayor a menor) para procesar las cartas más grandes primero
    valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)
    
    return valid_contours


def order_points(pts):
    """
    Ordena los 4 puntos de las esquinas de una carta en orden consistente:
    [top-left, top-right, bottom-right, bottom-left]
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Suma de coordenadas: top-left tendrá la menor suma, bottom-right la mayor
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    
    # Diferencia de coordenadas: top-right tendrá la menor diferencia, bottom-left la mayor
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    
    return rect


def get_card_corners(contour):
    """
    Obtiene las 4 esquinas de una carta a partir de su contorno.
    """
    # Calcular el perímetro del contorno
    perimeter = cv2.arcLength(contour, True)
    
    # Aproximar el contorno a un polígono
    # epsilon = 2% del perímetro: balance entre precisión y simplicidad
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Una carta debe tener exactamente 4 esquinas
    if len(approx) == 4:
        # Verificar que sea aproximadamente rectangular
        corners = approx.reshape(4, 2)
        ordered_corners = order_points(corners)
        
        # Calcular dimensiones
        width1 = np.linalg.norm(ordered_corners[0] - ordered_corners[1])
        width2 = np.linalg.norm(ordered_corners[2] - ordered_corners[3])
        height1 = np.linalg.norm(ordered_corners[0] - ordered_corners[3])
        height2 = np.linalg.norm(ordered_corners[1] - ordered_corners[2])
        
        avg_width = (width1 + width2) / 2
        avg_height = (height1 + height2) / 2
        
        # Asegurar que width < height (carta vertical)
        if avg_width > avg_height:
            avg_width, avg_height = avg_height, avg_width
            # Rotar los puntos 90 grados
            ordered_corners = np.array([
                ordered_corners[3],
                ordered_corners[0],
                ordered_corners[1],
                ordered_corners[2]
            ])
        
        # Verificar relación de aspecto
        if avg_height > 0:
            aspect_ratio = avg_width / avg_height
            if abs(aspect_ratio - CARD_ASPECT_RATIO) < CARD_ASPECT_TOLERANCE:
                return ordered_corners
    
    # Si no tiene 4 esquinas, intentar con el rectángulo mínimo
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype="float32")
    
    # Ordenar los puntos
    ordered_box = order_points(box)
    
    # Verificar dimensiones
    width = rect[1][0]
    height = rect[1][1]
    
    if width > height:
        width, height = height, width
        ordered_box = np.array([
            ordered_box[3],
            ordered_box[0],
            ordered_box[1],
            ordered_box[2]
        ])
    
    if height > 0:
        aspect_ratio = width / height
        if abs(aspect_ratio - CARD_ASPECT_RATIO) < CARD_ASPECT_TOLERANCE:
            return ordered_box
    
    return None


def perspective_transform(image, corners):
    """
    Aplica transformación de perspectiva para obtener una vista frontal de la carta.    
    """
    # Puntos de destino (carta normalizada)
    dst_points = np.array([
        [0, 0],
        [CARD_WIDTH - 1, 0],
        [CARD_WIDTH - 1, CARD_HEIGHT - 1],
        [0, CARD_HEIGHT - 1]
    ], dtype="float32")
    
    # Calcular matriz de transformación de perspectiva
    # getPerspectiveTransform: calcula la matriz 3x3 de transformación
    matrix = cv2.getPerspectiveTransform(corners, dst_points)
    
    # Aplicar la transformación
    # warpPerspective: aplica la transformación de perspectiva
    warped = cv2.warpPerspective(image, matrix, (CARD_WIDTH, CARD_HEIGHT))
    
    return warped


def ensure_correct_orientation(card_image):
    """
    Asegura que la carta esté en la orientación correcta (valor en esquina superior izquierda).
    Usa análisis de gradientes para detectar la orientación del texto.
    """
    # Convertir a escala de grises
    gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
    
    h, w = gray.shape
    
    # Extraer solo la esquina superior izquierda donde debería estar el valor
    corner_h = int(h * 0.30)
    corner_w = int(w * 0.20)
    
    # Probar las dos orientaciones posibles (0° y 180°)
    # Las cartas tienen el mismo contenido en esquinas opuestas pero invertido
    
    # Esquina superior izquierda
    tl_corner = gray[5:corner_h, 5:corner_w]
    
    # Esquina inferior derecha (rotada 180°)
    br_corner = gray[h-corner_h:h-5, w-corner_w:w-5]
    br_corner_rotated = cv2.rotate(br_corner, cv2.ROTATE_180)
    
    # Calcular el "peso" vertical de los píxeles oscuros
    # En una orientación correcta, el valor (número/letra) está arriba del palo
    # Por lo tanto, debería haber más contenido en la parte superior de la región
    
    def calculate_vertical_weight(corner_img):
        """Calcula si hay más contenido arriba (positivo) o abajo (negativo)"""
        _, thresh = cv2.threshold(corner_img, 140, 255, cv2.THRESH_BINARY_INV)
        
        h_c = thresh.shape[0]
        top_half = thresh[:h_c//2, :]
        bottom_half = thresh[h_c//2:, :]
        
        top_content = cv2.countNonZero(top_half)
        bottom_content = cv2.countNonZero(bottom_half)
        
        # Si hay más contenido arriba que abajo, probablemente está bien orientado
        return top_content - bottom_content
    
    # Calcular para ambas esquinas
    tl_weight = calculate_vertical_weight(tl_corner)
    br_weight = calculate_vertical_weight(br_corner_rotated)
    
    # También contar el total de contenido en cada esquina
    _, tl_thresh = cv2.threshold(tl_corner, 140, 255, cv2.THRESH_BINARY_INV)
    _, br_thresh = cv2.threshold(br_corner_rotated, 140, 255, cv2.THRESH_BINARY_INV)
    
    tl_total = cv2.countNonZero(tl_thresh)
    br_total = cv2.countNonZero(br_thresh)
    
    # Decisión: usar la esquina con más contenido Y mejor distribución vertical
    # Priorizar la que tenga más contenido en la parte superior
    tl_score = tl_total + tl_weight * 2
    br_score = br_total + br_weight * 2
    
    # Si la esquina inferior derecha tiene mejor score, rotar 180°
    if br_score > tl_score + 50:  # Margen de tolerancia
        return cv2.rotate(card_image, cv2.ROTATE_180)
    
    return card_image


def detect_cards(image, debug=False):
    """
    Función principal de detección de cartas.
    """
    # Preprocesar imagen
    blurred, hsv = preprocess_image(image)
    
    # Segmentar fondo verde
    mask = segment_green_background(hsv)
    
    # Encontrar contornos
    contours = find_card_contours(mask)
    
    # Procesar cada contorno
    cards = []
    for contour in contours:
        corners = get_card_corners(contour)
        
        if corners is not None:
            # Aplicar transformación de perspectiva
            card_image = perspective_transform(image, corners)
            
            # Asegurar orientación correcta
            card_image = ensure_correct_orientation(card_image)
            
            cards.append({
                'contour': contour,
                'corners': corners,
                'card_image': card_image
            })
    
    if debug:
        debug_info = {
            'blurred': blurred,
            'hsv': hsv,
            'mask': mask,
            'contours': contours
        }
        return cards, debug_info
    
    return cards


def draw_card_detection(image, cards, recognized_cards=None):
    """
    Dibuja la detección de cartas sobre la imagen.
    """
    output = image.copy()
    
    for i, card in enumerate(cards):
        corners = card['corners'].astype(np.int32)
        
        # Dibujar contorno de la carta
        cv2.polylines(output, [corners], True, (0, 255, 0), 3)
        
        # Dibujar esquinas
        for corner in corners:
            cv2.circle(output, tuple(corner), 8, (255, 0, 0), -1)
        
        # Si hay reconocimiento, mostrar el resultado
        if recognized_cards and i < len(recognized_cards):
            result = recognized_cards[i]
            if result:
                # Posición para el texto (encima de la carta)
                text_pos = (int(corners[0][0]), int(corners[0][1]) - 10)
                
                # Preparar texto (usar abreviaturas porque Unicode no se renderiza bien)
                value = result.get('value', '?')
                suit = result.get('suit', '?')
                
                # Abreviaturas de palos
                suit_names = {
                    'picas': 'PICAS', 'corazones': 'CORAZ', 
                    'diamantes': 'DIAM', 'treboles': 'TREB'
                }
                suit_text = suit_names.get(suit, '?') if suit else '?'
                text = f"{value} {suit_text}"
                
                # Dibujar fondo para el texto
                (text_width, text_height), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2
                )
                cv2.rectangle(
                    output,
                    (text_pos[0] - 5, text_pos[1] - text_height - 5),
                    (text_pos[0] + text_width + 5, text_pos[1] + 5),
                    (255, 255, 255),
                    -1
                )
                
                # Dibujar texto
                color = (0, 0, 255) if suit in ['corazones', 'diamantes'] else (0, 0, 0)
                cv2.putText(
                    output, text, text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
                )
    
    return output

