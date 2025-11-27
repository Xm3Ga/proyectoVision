"""
Controles:
- ESC o 'q': Salir
- 's': Capturar imagen actual
- 'd': Mostrar/ocultar información de depuración
- 'c': Calibrar colores del tapete verde
- 'r': Reiniciar reconocedor
"""

import cv2
import numpy as np
import os
from datetime import datetime

from config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT,
    SHOW_DEBUG_WINDOWS, CARD_WIDTH, CARD_HEIGHT
)
from card_detector import detect_cards, draw_card_detection
from card_recognizer_v3 import CardRecognizerV3


class CardRecognitionSystem:
    """
    Sistema principal de reconocimiento de cartas.
    """
    
    def __init__(self):
        """Inicializa el sistema."""
        self.cap = None
        self.recognizer = None
        self.show_debug = SHOW_DEBUG_WINDOWS
        self.capture_count = 0
        
        # Crear directorio para capturas
        os.makedirs('capturas', exist_ok=True)
    
    def initialize(self):
        """
        Inicializa la cámara y el reconocedor.
        """
        print("=" * 60)
        print("   SISTEMA DE RECONOCIMIENTO DE CARTAS DE PÓKER")
        print("=" * 60)
        
        # Verificar plantillas de cartas completas
        cards_path = 'templates/cards'
        if not os.path.exists(cards_path) or len(os.listdir(cards_path)) == 0:
            print("\n⚠ No hay plantillas de cartas.")
            print("  Ejecuta 'python capture_cards.py' para capturar las cartas.")
        
        # Inicializar reconocedor v3 (diferencia absoluta - mejor precisión)
        print("\nCargando reconocedor de cartas...")
        self.recognizer = CardRecognizerV3()
        
        # Inicializar cámara
        print(f"\nIniciando cámara (índice {CAMERA_INDEX})...")
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        
        if not self.cap.isOpened():
            print("✗ Error: No se pudo abrir la cámara")
            print("  Verifica que Camo esté conectado y funcionando")
            return False
        
        # Configurar resolución
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        
        # Obtener resolución real
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"✓ Cámara inicializada: {actual_width}x{actual_height}")
        print("\nControles:")
        print("  ESC/'q': Salir")
        print("  's': Capturar imagen")
        print("  'd': Toggle depuración")
        print("  'c': Calibrar color verde")
        print("\n¡Sistema listo!")
        
        return True
    
    def run(self):
        """
        Bucle principal del sistema.
        """
        if not self.initialize():
            return
        
        print("\nProcesando video en tiempo real...")
        
        fps_start_time = cv2.getTickCount()
        fps_frame_count = 0
        fps = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error leyendo frame")
                break
            
            # Calcular FPS
            fps_frame_count += 1
            if fps_frame_count >= 10:
                fps_end_time = cv2.getTickCount()
                fps = 10 / ((fps_end_time - fps_start_time) / cv2.getTickFrequency())
                fps_start_time = fps_end_time
                fps_frame_count = 0
            
            # Detectar cartas
            if self.show_debug:
                cards, debug_info = detect_cards(frame, debug=True)
            else:
                cards = detect_cards(frame)
                debug_info = None
            
            # Reconocer cartas
            recognized = []
            if cards and self.recognizer:
                recognized = self.recognizer.recognize_cards(cards)
            
            # Dibujar resultados
            output = draw_card_detection(frame, cards, recognized)
            
            # Mostrar información
            self.draw_info(output, fps, len(cards), recognized)
            
            # Mostrar ventana principal
            cv2.imshow('Reconocimiento de Cartas', output)
            
            # Mostrar ventanas de depuración
            if self.show_debug and debug_info:
                self.show_debug_windows(debug_info, cards)
            
            # Manejar teclas
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC o 'q'
                break
            elif key == ord('s'):
                self.save_capture(frame, cards, recognized)
            elif key == ord('d'):
                self.show_debug = not self.show_debug
                if not self.show_debug:
                    cv2.destroyWindow('Máscara')
                    cv2.destroyWindow('Cartas Detectadas')
            elif key == ord('c'):
                self.calibrate_green(frame)
        
        self.cleanup()
    
    def draw_info(self, image, fps, num_cards, recognized):
        """Dibuja información en la imagen."""
        h, w = image.shape[:2]
        
        # Panel de información semi-transparente
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (350, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        
        # Texto
        cv2.putText(image, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(image, f"Cartas detectadas: {num_cards}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Mostrar cartas reconocidas
        if recognized:
            cards_text = []
            for r in recognized:
                value = r.get('value')
                suit = r.get('suit')
                conf = r.get('confidence', 0)
                
                # Usar abreviaturas para los palos (Unicode no se renderiza bien en OpenCV)
                suit_abbrev = {
                    'picas': 'P', 'corazones': 'C', 
                    'diamantes': 'D', 'treboles': 'T'
                }
                
                if value and suit:
                    abbr = suit_abbrev.get(suit, '?')
                    cards_text.append(f"{value}-{abbr} ({conf:.0%})")
                elif value:
                    cards_text.append(f"{value}-? ({conf:.0%})")
                else:
                    cards_text.append("???")
            
            text = "Cartas: " + ", ".join(cards_text)
            cv2.putText(image, text, (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Controles
        cv2.putText(image, "ESC: Salir | s: Capturar | d: Debug", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def show_debug_windows(self, debug_info, cards):
        """Muestra ventanas de depuración."""
        # Máscara de segmentación
        mask_display = cv2.cvtColor(debug_info['mask'], cv2.COLOR_GRAY2BGR)
        
        # Dibujar contornos en la máscara
        cv2.drawContours(mask_display, debug_info['contours'], -1, (0, 255, 0), 2)
        
        cv2.imshow('Máscara', mask_display)
        
        # Cartas detectadas
        if cards:
            # Crear mosaico de cartas
            card_images = [card['card_image'] for card in cards[:4]]  # Máximo 4 cartas
            
            if card_images:
                # Redimensionar todas al mismo tamaño
                resized = [cv2.resize(img, (CARD_WIDTH, CARD_HEIGHT)) for img in card_images]
                
                # Crear grid
                if len(resized) == 1:
                    mosaic = resized[0]
                elif len(resized) == 2:
                    mosaic = np.hstack(resized)
                else:
                    # Completar con imágenes negras si es necesario
                    while len(resized) < 4:
                        resized.append(np.zeros((CARD_HEIGHT, CARD_WIDTH, 3), dtype=np.uint8))
                    top = np.hstack(resized[:2])
                    bottom = np.hstack(resized[2:4])
                    mosaic = np.vstack([top, bottom])
                
                cv2.imshow('Cartas Detectadas', mosaic)
    
    def save_capture(self, frame, cards, recognized):
        """Guarda una captura de la imagen actual."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.capture_count += 1
        
        # Guardar frame original
        filename = f"capturas/captura_{timestamp}_{self.capture_count}.png"
        cv2.imwrite(filename, frame)
        print(f"✓ Captura guardada: {filename}")
        
        # Guardar cartas individuales
        for i, card in enumerate(cards):
            card_filename = f"capturas/carta_{timestamp}_{self.capture_count}_{i}.png"
            cv2.imwrite(card_filename, card['card_image'])
            
            if i < len(recognized):
                r = recognized[i]
                if r['value'] and r['suit']:
                    print(f"  Carta {i+1}: {r['value']} de {r['suit']}")
    
    def calibrate_green(self, frame):
        """
        Permite calibrar el rango de color verde del tapete.
        """
        from config import GREEN_HSV_LOWER, GREEN_HSV_UPPER
        import config
        
        print("\n=== Calibración del color verde ===")
        print("Usa los trackbars para ajustar el rango HSV")
        print("Presiona 'g' para guardar o 'c' para cancelar")
        
        cv2.namedWindow('Calibración')
        
        # Crear trackbars
        cv2.createTrackbar('H min', 'Calibración', GREEN_HSV_LOWER[0], 180, lambda x: None)
        cv2.createTrackbar('S min', 'Calibración', GREEN_HSV_LOWER[1], 255, lambda x: None)
        cv2.createTrackbar('V min', 'Calibración', GREEN_HSV_LOWER[2], 255, lambda x: None)
        cv2.createTrackbar('H max', 'Calibración', GREEN_HSV_UPPER[0], 180, lambda x: None)
        cv2.createTrackbar('S max', 'Calibración', GREEN_HSV_UPPER[1], 255, lambda x: None)
        cv2.createTrackbar('V max', 'Calibración', GREEN_HSV_UPPER[2], 255, lambda x: None)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Obtener valores actuales
            h_min = cv2.getTrackbarPos('H min', 'Calibración')
            s_min = cv2.getTrackbarPos('S min', 'Calibración')
            v_min = cv2.getTrackbarPos('V min', 'Calibración')
            h_max = cv2.getTrackbarPos('H max', 'Calibración')
            s_max = cv2.getTrackbarPos('S max', 'Calibración')
            v_max = cv2.getTrackbarPos('V max', 'Calibración')
            
            # Aplicar máscara
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (h_min, s_min, v_min), (h_max, s_max, v_max))
            
            # Mostrar resultado
            result = cv2.bitwise_and(frame, frame, mask=mask)
            combined = np.hstack([frame, result])
            combined = cv2.resize(combined, (1280, 360))
            
            cv2.imshow('Calibración', combined)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('g'):
                # Guardar valores
                config.GREEN_HSV_LOWER = (h_min, s_min, v_min)
                config.GREEN_HSV_UPPER = (h_max, s_max, v_max)
                print(f"✓ Valores guardados:")
                print(f"  GREEN_HSV_LOWER = ({h_min}, {s_min}, {v_min})")
                print(f"  GREEN_HSV_UPPER = ({h_max}, {s_max}, {v_max})")
                break
            elif key == ord('c') or key == 27:
                print("Calibración cancelada")
                break
        
        cv2.destroyWindow('Calibración')
    
    def cleanup(self):
        """Libera recursos."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Sistema cerrado correctamente")


def process_single_image(image_path):
    """
    Procesa una imagen individual (útil para pruebas).
    
    Parámetros:
    - image_path: Ruta a la imagen
    """
    # Cargar imagen
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: No se pudo cargar la imagen '{image_path}'")
        return
    
    # Inicializar reconocedor v3
    recognizer = CardRecognizerV3()
    
    # Detectar cartas
    cards, debug_info = detect_cards(image, debug=True)
    
    # Reconocer
    recognized = recognizer.recognize_cards(cards)
    
    # Mostrar resultados
    print(f"\nCartas detectadas: {len(cards)}")
    for i, (card, result) in enumerate(zip(cards, recognized)):
        value = result['value'] or '?'
        suit = result['suit'] or '?'
        symbol = result['suit_symbol']
        confidence = result.get('confidence', 0)
        print(f"  Carta {i+1}: {value} de {suit} ({symbol})")
        print(f"    Confianza: {confidence:.2%}")
    
    # Mostrar imagen con resultados
    output = draw_card_detection(image, cards, recognized)
    
    cv2.imshow('Resultado', output)
    cv2.imshow('Máscara', debug_info['mask'])
    
    if cards:
        cv2.imshow('Primera carta', cards[0]['card_image'])
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """Función principal."""
    import sys
    
    if len(sys.argv) > 1:
        # Procesar imagen individual
        image_path = sys.argv[1]
        process_single_image(image_path)
    else:
        # Modo tiempo real
        system = CardRecognitionSystem()
        system.run()


if __name__ == "__main__":
    main()

