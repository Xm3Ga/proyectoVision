import cv2
import numpy as np
import os
from config import CARD_VALUES, CARD_SUITS, SUIT_SYMBOLS, CARD_WIDTH, CARD_HEIGHT


class CardRecognizerV3:
    """
    Reconocedor usando diferencia absoluta de píxeles.
    Compara imágenes binarizadas - menor diferencia = mejor match.
    """
    
    def __init__(self, templates_path='templates/cards'):
        """Inicializa el reconocedor cargando las plantillas."""
        self.templates_path = templates_path
        self.card_templates = {}  # {(valor, palo): imagen_binarizada}
        self.templates_loaded = False
        
        self.load_templates()
    
    def load_templates(self):
        """Carga todas las plantillas de cartas."""
        os.makedirs(self.templates_path, exist_ok=True)
        
        count = 0
        for value in CARD_VALUES:
            for suit in CARD_SUITS:
                filename = f"{value}_{suit}.png"
                filepath = os.path.join(self.templates_path, filename)
                
                if os.path.exists(filepath):
                    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        # Redimensionar al tamaño estándar
                        img = cv2.resize(img, (CARD_WIDTH, CARD_HEIGHT))
                        # Umbralizar para obtener imagen binaria
                        _, img_thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
                        self.card_templates[(value, suit)] = img_thresh
                        count += 1
        
        self.templates_loaded = count > 0
        
        if count > 0:
            print(f"✓ Plantillas cargadas: {count}/52")
        else:
            print("⚠ No hay plantillas. Ejecuta 'python capture_cards.py' primero.")
    
    def preprocess_card(self, card_image):
        """
        Preprocesa la carta para comparación.
        Convierte a escala de grises, redimensiona y umbraliza.
        """
        # Convertir a escala de grises si es necesario
        if len(card_image.shape) == 3:
            gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = card_image.copy()
        
        # Redimensionar al tamaño estándar
        gray = cv2.resize(gray, (CARD_WIDTH, CARD_HEIGHT))
        
        # Umbralizar a binario
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        return thresh
    
    def compare_cards(self, card_thresh, template_thresh):
        """
        Compara dos imágenes binarizadas usando diferencia absoluta.
        Retorna la suma de diferencias (menor = más similar).
        """
        diff = cv2.absdiff(card_thresh, template_thresh)
        score = np.sum(diff)
        return score
    
    def recognize_card(self, card_image):
        """
        Reconoce una carta comparándola con todas las plantillas.
        Prueba tanto orientación normal como rotada 180°.
        """
        if not self.card_templates:
            return {
                'value': None,
                'suit': None,
                'confidence': 0,
                'suit_symbol': '?'
            }
        
        # Preprocesar la carta de entrada
        card_thresh = self.preprocess_card(card_image)
        
        # También preparar versión rotada 180°
        card_thresh_rotated = cv2.rotate(card_thresh, cv2.ROTATE_180)
        
        best_match = None
        min_diff = float('inf')
        
        # Comparar con cada plantilla
        for (value, suit), template in self.card_templates.items():
            # Comparar orientación normal
            diff_normal = self.compare_cards(card_thresh, template)
            
            # Comparar orientación rotada
            diff_rotated = self.compare_cards(card_thresh_rotated, template)
            
            # Usar el mejor de los dos
            diff = min(diff_normal, diff_rotated)
            
            if diff < min_diff:
                min_diff = diff
                best_match = (value, suit)
        
        if best_match:
            value, suit = best_match
            
            # Calcular confianza (invertir: menor diff = mayor confianza)
            # Normalizar aproximadamente entre 0 y 1
            max_possible_diff = CARD_WIDTH * CARD_HEIGHT * 255
            confidence = 1.0 - (min_diff / max_possible_diff)
            
            return {
                'value': value,
                'suit': suit,
                'confidence': confidence,
                'suit_symbol': SUIT_SYMBOLS.get(suit, '?'),
                'diff_score': min_diff
            }
        
        return {
            'value': None,
            'suit': None,
            'confidence': 0,
            'suit_symbol': '?'
        }
    
    def recognize_cards(self, cards):
        """Reconoce múltiples cartas."""
        results = []
        for card in cards:
            result = self.recognize_card(card['card_image'])
            results.append(result)
        return results

