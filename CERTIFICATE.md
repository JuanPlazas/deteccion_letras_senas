# Certificado de Autoría y Publicación de Código Abierto

**Proyecto:** Detección de Letras en Lengua de Señas Colombiana (LSC) — pipeline de captura, entrenamiento y transcripción en tiempo real.

**Autor:** JuanPlazas

**Fecha:** 21 de septiembre de 2026

---

Por la presente se certifica que el proyecto *Detección de Letras en Lengua de Señas Colombiana (LSC)* — incluyendo su diseño, implementación, dataset y documentación — fue desarrollado íntegramente por **JuanPlazas**, quien declara ser su autor.

## Contenido que respalda esta autoría

- **Pipeline de captura de datos**: secuencias normalizadas de 21 landmarks de la mano mediante MediaPipe (`scripts/capture_data.py`, `utils/hands_detector.py`).
- **Dataset público de uso abierto**: alfabeto dactilológico A–Z + Ñ, ~838 secuencias de forma `(30, 42)`, representadas en 2 dimensiones (x, y) y normalizadas respecto a la muñeca.
- **Pipeline de entrenamiento**: modelo LSTM de dos capas con seguimiento de experimentos en MLflow (`training/`).
- **Aplicación de transcripción en tiempo real**: interfaz limpia sin overlays de esqueleto, subtitulado inferior, barra de progreso de frames y deduplicación de señas (`app.py`).

## Declaración de publicación

El autor **publica este proyecto como código abierto** y otorga permiso para su uso, ejecución, copia, modificación y distribución con fines educativos, académicos, de investigación o de desarrollo, siempre que se **cite la autoría** del trabajo original.

Este certificado se emite como declaración de autoría y no reemplaza un archivo de licencia formal (por ejemplo, MIT, Apache-2.0) que pueda añadirse al repositorio para regular términos legales específicos.

---

**Firma:** JuanPlazas

**Fecha de emisión:** 21 de septiembre de 2026