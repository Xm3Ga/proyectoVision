# 🃏 Sistema de Reconocimiento de Cartas de Póker

Sistema de visión artificial para reconocer cartas de póker en tiempo real usando **técnicas clásicas de procesamiento de imágenes** (sin redes neuronales ni aprendizaje automático).

## 📋 Descripción

Este proyecto detecta e identifica cartas de una baraja estándar de póker (52 cartas) colocadas sobre un tapete verde, reconociendo tanto el **valor** (A, 2-10, J, Q, K) como el **palo** (♠ ♥ ♦ ♣).

### Técnicas utilizadas:
- Segmentación por color HSV
- Detección de contornos
- Transformación de perspectiva
- Comparación por diferencia absoluta de píxeles

## 🔧 Requisitos

- Python 3.8+
- OpenCV
- NumPy
- Cámara (webcam o smartphone via Camo)
- Tapete verde

## 📦 Instalación

```bash
git clone https://github.com/Xm3Ga/proyectoVision.git
cd proyectoVision
pip install -r requirements.txt
```

## 🚀 Uso

### 1. Capturar plantillas (primera vez)
```bash
python capture_cards.py
```
Coloca cada carta sobre el tapete y presiona las teclas correspondientes para guardarla.

### 2. Ejecutar reconocimiento
```bash
python main.py
```

### Controles
| Tecla | Acción |
|-------|--------|
| ESC | Salir |
| S | Capturar imagen |
| D | Toggle depuración |
| C | Calibrar color verde |

## 📁 Estructura

```
proyectoVision/
├── main.py              # Programa principal
├── card_detector.py     # Detección de cartas
├── card_recognizer_v3.py # Reconocimiento
├── config.py            # Configuración
├── capture_cards.py     # Capturador de plantillas
├── requirements.txt     # Dependencias
└── templates/cards/     # 52 plantillas
```

## 📄 Documentación

Ver [MEMORIA_TECNICA.md](MEMORIA_TECNICA.md) para detalles técnicos completos.

## 👤 Autor

**Adrian Pérez Bahamontes**

Proyecto desarrollado para el Examen Parcial de Inteligencia Artificial - Noviembre 2025

