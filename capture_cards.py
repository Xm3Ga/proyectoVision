"""
Capturador de plantillas de cartas completas.

Instrucciones:
1. Coloca una carta sobre el tapete verde
2. Presiona la tecla del VALOR (A, 2-9, 0=10, J, Q, K)
3. Luego presiona la tecla del PALO (H, D, S, C)
4. La carta se guarda automáticamente
5. ESC para salir

Teclas de palos:
  H = Hearts (Corazones) ♥
  D = Diamonds (Diamantes) ♦  
  S = Spades (Picas) ♠
  C = Clubs (Tréboles) ♣
"""

import cv2
import numpy as np
import os
from config import (
    CAMERA_INDEX, CARD_WIDTH, CARD_HEIGHT,
    CARD_VALUES, CARD_SUITS, SUIT_SYMBOLS
)
from card_detector import detect_cards


def main():
    # Crear directorio
    templates_path = 'templates/cards'
    os.makedirs(templates_path, exist_ok=True)
    
    print("=" * 60)
    print("   CAPTURADOR DE CARTAS COMPLETAS")
    print("=" * 60)
    print("\nPaso 1: Coloca una carta sobre el tapete verde")
    print("Paso 2: Presiona tecla del VALOR: A,2,3,4,5,6,7,8,9,0(=10),J,Q,K")
    print("Paso 3: Presiona tecla del PALO: H(♥), D(♦), S(♠), C(♣)")
    print("\nOtras teclas:")
    print("  R = Rotar carta 180°")
    print("  ESC = Salir")
    print("-" * 60)
    
    # Mapeo de teclas
    value_keys = {
        ord('a'): 'A', ord('A'): 'A',
        ord('2'): '2', ord('3'): '3', ord('4'): '4', ord('5'): '5',
        ord('6'): '6', ord('7'): '7', ord('8'): '8', ord('9'): '9',
        ord('0'): '10',
        ord('j'): 'J', ord('J'): 'J',
        ord('q'): 'Q', ord('Q'): 'Q',
        ord('k'): 'K', ord('K'): 'K',
    }
    
    suit_keys = {
        ord('h'): 'corazones', ord('H'): 'corazones',
        ord('d'): 'diamantes', ord('D'): 'diamantes',
        ord('s'): 'picas', ord('S'): 'picas',
        ord('c'): 'treboles', ord('C'): 'treboles',
    }
    
    # Estado
    pending_value = None
    manual_rotation = False
    
    # Contar cartas existentes
    existing = sum(1 for v in CARD_VALUES for s in CARD_SUITS 
                   if os.path.exists(os.path.join(templates_path, f"{v}_{s}.png")))
    
    print(f"\nCartas capturadas: {existing}/52")
    
    # Abrir cámara
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("✗ Error: No se pudo abrir la cámara")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("✓ Cámara iniciada")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detectar cartas
        cards = detect_cards(frame)
        
        # Visualización
        display = frame.copy()
        current_card = None
        
        for card in cards:
            corners = card['corners'].astype(np.int32)
            cv2.polylines(display, [corners], True, (0, 255, 0), 3)
            current_card = card['card_image'].copy()
        
        # Aplicar rotación manual
        if current_card is not None and manual_rotation:
            current_card = cv2.rotate(current_card, cv2.ROTATE_180)
        
        # Mostrar carta detectada
        if current_card is not None:
            card_display = current_card.copy()
            if manual_rotation:
                cv2.putText(card_display, "ROTADA", (5, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imshow('Carta Detectada', card_display)
        
        # Panel de información
        info_h = 150
        info_panel = np.zeros((info_h, 400, 3), dtype=np.uint8)
        
        # Progreso
        cv2.putText(info_panel, f"Cartas: {existing}/52", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Estado actual
        if pending_value:
            cv2.putText(info_panel, f"Valor: {pending_value} - Ahora pulsa PALO", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(info_panel, "H=corazones D=diamantes S=picas C=treboles", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            cv2.putText(info_panel, "Pulsa VALOR: A,2-9,0(10),J,Q,K", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Rotación
        rot_text = "R=Rotar (ON)" if manual_rotation else "R=Rotar (OFF)"
        cv2.putText(info_panel, rot_text, (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        cv2.putText(info_panel, "ESC=Salir", (10, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        cv2.imshow('Info', info_panel)
        
        # Info en frame principal
        cv2.putText(display, f"Cartas: {existing}/52", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if pending_value:
            cv2.putText(display, f"Valor: {pending_value} -> Pulsa PALO", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow('Camara', display)
        
        # Teclas
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        
        elif key == ord('r') or key == ord('R'):
            manual_rotation = not manual_rotation
            print(f"  Rotación: {'ON' if manual_rotation else 'OFF'}")
        
        elif key in value_keys:
            pending_value = value_keys[key]
            print(f"  Valor seleccionado: {pending_value}")
        
        elif key in suit_keys and pending_value and current_card is not None:
            suit = suit_keys[key]
            
            # Guardar carta
            filename = f"{pending_value}_{suit}.png"
            filepath = os.path.join(templates_path, filename)
            
            # Convertir a escala de grises y guardar
            if len(current_card.shape) == 3:
                card_gray = cv2.cvtColor(current_card, cv2.COLOR_BGR2GRAY)
            else:
                card_gray = current_card
            
            cv2.imwrite(filepath, card_gray)
            
            symbol = SUIT_SYMBOLS.get(suit, '?')
            print(f"  ✓ Guardada: {pending_value} de {suit} ({symbol})")
            
            existing += 1
            pending_value = None
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n{'='*60}")
    print(f"   RESUMEN: {existing}/52 cartas capturadas")
    print(f"{'='*60}")
    
    # Mostrar cartas faltantes
    missing = []
    for value in CARD_VALUES:
        for suit in CARD_SUITS:
            if not os.path.exists(os.path.join(templates_path, f"{value}_{suit}.png")):
                missing.append(f"{value}{SUIT_SYMBOLS[suit]}")
    
    if missing:
        print(f"\nCartas faltantes ({len(missing)}):")
        # Mostrar en filas de 13
        for i in range(0, len(missing), 13):
            print("  " + " ".join(missing[i:i+13]))
    else:
        print("\n✓ ¡Todas las cartas capturadas!")


if __name__ == "__main__":
    main()

