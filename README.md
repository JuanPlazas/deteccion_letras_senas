# Detección de Letras en Lengua de Señas Colombiana (LSC)

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow%202.21-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV%205.0-5C3EE8?logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-09B3AF?logo=google&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![uv](https://img.shields.io/badge/uv-DE5FE9?logo=uv&logoColor=white)

Sistema de **detección y transcripción en tiempo real** de las letras del alfabeto dactilológico de la Lengua de Señas Colombiana (LSC), desarrollado como proyecto de portafolio.

El proyecto cubre el pipeline completo:

1. **Captura de datos** — grabación de secuencias de _landmarks_ de la mano con MediaPipe.
2. **Entrenamiento** — modelo de **LSTM (Keras)** con seguimiento de experimentos en **MLflow**.
3. **Inferencia** — **aplicación en tiempo real** (`app.py`) que transcribe las señas como texto en pantalla.

---

## Estado del proyecto (qué quedó listo)

### Dataset público

- **27 clases**: A–Z **+ Ñ** del dactilológico colombiano.
- **~838 secuencias** de entrenamiento, cada una de forma `(30, 42)`: 30 frames × 21 _landmarks_ × 2 coordenadas (x, y).
- Publicado como **dato de uso abierto**: al no existir un dataset de lengua de señas colombiana disponible públicamente para este fin, el dataset se genera y publica junto con el proyecto para que la comunidad pueda usarlo libremente (docencia, investigación y desarrollo).

### Modelo entrenado

- Arquitectura **LSTM de dos capas** (64 → 32 unidades, dropout 0.3) + capa densa softmax.
- Último run con seguimiento en MLflow: **~92.9% de exactitud en el split de prueba**.
- Cada entrenamiento guarda en MLflow: hiperparámetros, métricas por época, el modelo con firma y el _scaler_ de preprocesamiento.

### Aplicación real-time (`app.py`)

Interfaz limpia pensada para grabar demos (por ejemplo, deletrear la palabra "HOLA"):

- **No dibuja las conexiones ni el esqueleto de la mano**: la vista es el video natural en espejo.
- Barra de **progreso de frames** en la parte superior mientras se llena la ventana de captura.
- Al detectar una seña, **escribe la letra en la parte inferior de la pantalla** (banner de subtítulos).
- **Deduplicación**: si vuelve a aparecer la misma letra que la última detectada, no se agrega.
- **Tecla `X` borra todas las letras** detectadas hasta el momento.
- Controles adicionales: `ESPACIO` separa palabras, `RETROCESO` borra la última letra, `Q`/`ESC` salen.

---

## Arquitectura: cómo evolucionó la solución

> **El proyecto se inició pensando en usar YOLO con OpenCV**, enfocado en detectar y clasificar las manos sobre imágenes de video.

Durante el análisis se identificó un punto clave de la **Lengua de Señas Colombiana**: varias señas (incluso muchas de sus letras) son **dinámicas, implican movimiento** en el tiempo. Clasificar frame a frame con YOLO pierde esa información temporal, por lo que **se decidió cambiar a una arquitectura basada en LSTM**, que modela secuencias temporales completas.

Con ese cambio, la representación de los datos también cambió:

- **El dataset ya no son imágenes**: cada muestra es la secuencia de los 21 puntos de la mano (**landmarks**) que obtiene **MediaPipe**.
- Cada frame aporta 42 valores (**solo 2 dimensiones: x, y**); no se usan los 3 ejes (sin coordenada z).
- Los landmarks se **normalizan** tomando la **muñeca (landmark 0) como punto central** y dividiendo las coordenadas por la **distancia de la muñeca al nudillo del dedo medio (landmark 9)**. Esto hace al modelo invariante a la posición de la mano en el cuadro y a su distancia a la cámara.
- El eje x se **espeja cuando la mano es la izquierda**, para aprender un único patrón con cualquiera de las dos manos.

### Resultado

| Enfoque          | Representación             | Problema detectado                       | Decisión         |
| ---------------- | -------------------------- | ---------------------------------------- | ---------------- |
| YOLO + OpenCV    | Imágenes, frame a frame    | No captura señas con movimiento          | Descartado       |
| **LSTM (Keras)** | Secuencias de 21 landmarks | Permite modelar el movimiento y la forma | **Implementado** |

---

## Estructura del repositorio

```text
deteccion_letras_senas/
├── app.py                     # Aplicación de transcripción en tiempo real
├── scripts/
│   └── capture_data.py        # Captura de secuencias de landmarks (.npy)
├── training/
│   ├── config.py              # Hiperparámetros y ajustes de MLflow
│   ├── data_loader.py         # Carga del dataset, split y scaler
│   ├── model.py               # Arquitectura del modelo LSTM
│   └── train.py               # Entrenamiento y registro en MLflow
├── utils/
│   ├── config.py              # Constantes y rutas compartidas
│   └── hands_detector.py      # Wrapper de MediaPipe (21 landmarks + normalización)
├── data/sequences/            # Dataset público (A–Z + Ñ), shape (30, 42)
├── models/                    # Modelo hand_landmarker.task (descarga automática)
├── pyproject.toml             # Configuración del proyecto (uv)
├── uv.lock                    # Dependencias bloqueadas
└── README.md
```

---

## Requisitos

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- Cámara web

## Instalación

```bash
uv sync
```

---

## Comandos de uso

Ejecutar siempre desde la raíz del proyecto.

### 1. Entrenamiento

```bash
uv run python -m training.train
```

El script carga el dataset, entrena el LSTM y registra todo en MLflow (parámetros, métricas por época, modelo y preprocesamiento).

### 2. Ver los experimentos en la UI de MLflow

```bash
uv run mlflow server --backend-store-uri sqlite:///<RUTA_ABSOLUTA_AL_PROYECTO>/training/mlflow.db
```

Abrir en el navegador: <http://127.0.0.1:5000>.

> **Importante (Windows):** la URI debe apuntar explícitamente a la base que usa el entrenamiento (`training/mlflow.db`) y usar **barras normales `/`**. Si se omite `--backend-store-uri`, MLflow crea un store nuevo basado en archivos (`./mlruns`) y la interfaz aparecerá vacía.

### 3. Aplicación de transcripción en tiempo real

```bash
uv run app.py
```

| Argumento    | Descripción                                                   |
| ------------ | ------------------------------------------------------------- |
| `--camera`   | Índice de la cámara web (por defecto `0`).                    |
| `--min-conf` | Confianza mínima para aceptar una letra (por defecto `0.55`). |

> Si la ventana se cierra al instante, prueba otros índices de cámara (`--camera 1`, `2`, …).

### 4. Capturar nuevas secuencias (opcional)

```bash
uv run scripts/capture_data.py --letter S
```

---

## Controles de la app

| Tecla       | Acción                              |
| ----------- | ----------------------------------- |
| `X`         | Borra todas las letras detectadas   |
| `ESPACIO`   | Agrega un espacio (separa palabras) |
| `RETROCESO` | Borra la última letra               |
| `Q` / `ESC` | Salir                               |

---

## Notas de diseño

- Los landmarks se normalizan usando la **muñeca (landmark 0) como origen** y dividiendo por la **distancia muñeca → nudillo medio (landmark 9)**: el modelo es invariante a posición y a distancia de la cámara.
- Las coordenadas **x se espejan para mano izquierda**, de modo que el modelo generaliza con cualquiera de las dos manos.
- Si la mano se pierde a mitad de la secuencia, esta se descarta para mantener el dataset limpio.
- La aplicación acumula frames hasta completar la ventana (30 frames) y solo entonces predice; las letras repetidas consecutivas no se agregan al texto.

---

## Certificado y licencia

Este proyecto es **de autoría propia** y se publica como **código abierto**. Ver [`CERTIFICATE.md`](CERTIFICATE.md).
