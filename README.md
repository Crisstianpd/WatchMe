# WatchMe
*Detección de rostros y emociones con MediaPipe y el sistema FACS.*
<p align="center">
    <img src="./assets/watchme.png" width="500px">
</p>



## ¿Qué es WatchMe?

**WatchMe** es un prototipo desarrollado en **Python** que permite la **detección de rostros** y la **identificación de emociones** de forma **eficiente y ligera**, sin depender de modelos de aprendizaje profundo pesados que consuman grandes recursos computacionales.

El programa utiliza:

- **[MediaPipe Face Mesh](https://chuoling.github.io/mediapipe/)** > para la detección y mapeo de 468 puntos faciales (landmarks) en 2D.
<p align="center">
    <img src="https://images.viblo.asia/d70d57f3-6756-47cd-a942-249cc1a7da82.png" height="80px">
</p>

- **[OpenCV](https://opencv.org/)** > para la captura de video desde la cámara, manipulación de frames y visualización.
<p align="center">
    <img src="https://www.nxrte.com/wp-content/uploads/2024/06/opencv.webp" height="80px">
</p>


## Sistema de Detección de Emociones: **FACS**

### ¿Qué es FACS?

El **Facial Action Coding System (FACS)** es un estándar desarrollado por los psicólogos **Paul Ekman** y **Wallace V. Friesen** en 1978. Codifica **microexpresiones faciales** (llamadas *Action Units* o **AU**) que, en combinación, forman emociones reconocibles.

> Ejemplo:  
> `AU6 + AU12` > Elevación de mejillas + comisuras hacia arriba = **Felicidad**

### ¿Por qué FACS en lugar de redes neuronales?

Se decidió utilizar el sistema **FACS** ya que permite identificar de forma más óptima y explicable microexpresiones faciales a partir de los **landmarks** detectados por MediaPipe.  
Estas microexpresiones, evaluadas en conjunto, permiten inferir la emoción del rostro detectado **sin necesidad de redes neuronales profundas**, lo que reduce significativamente el consumo de recursos y permite ejecutar el sistema en computadoras modestas.



## Funcionamiento del Sistema FACS en WatchMe

### Paso 1: Detección del rostro con MediaPipe
Se procesa el frame en RGB y es pasado a MediaPipe para que devuelva los **468 landmarks** para el rostro detectado.

### Paso 2: Normalización espacial (para precisión)
Debido a la inconsistencia de los datos dependiendo de la distancia o ángulo de inclinación del rostro detectado, se implementa:

#### Recorte dinámico del rostro
```python
face_crop = frame[y_min:y_max, x_min:x_max]
```
> Se extrae solo la región del rostro.

#### Redimensionado a lienzo fijo de **200x200**
```python
target_size = 200
scale = min(target_size / face_w, target_size / face_h)
resized_face = cv2.resize(face_crop, (new_w, new_h))
```

#### Normalización con distancia interocular
```python
...

def normalized_distance(p1: tuple[int, int], p2: tuple[int, int], ref_dist: float) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1]) / ref_dist * 100

...

# Dato para normalizar puntos (distancia interocular)
ref_dist = math.hypot(points[263][0] - points[33][0], points[263][1] - points[33][1])

```
> Todos los valores se expresan como **porcentaje relativo a la distancia entre ojos**, eliminando efectos de escala.

```python
metrics = {
    ...
    "ceja_izq": normalized_distance(points[295], points[385], ref_dist),
    "ancho_boca": normalized_distance(points[78], points[308], ref_dist),
    "alto_boca": normalized_distance(points[13], points[14], ref_dist),
    ...
}
```

---

### Paso 3: Cálculo de microexpresiones (Action Units)

Se miden distancias entre **puntos clave** normalizados, tomando como base las siguientes métricas

| Métrica                 | Puntos MediaPipe | Significado                    | AU (Action Unit)             |
| ----------------------- | ---------------- | ------------------------------ | ---------------------------- |
| `ceja_der`              | 65 → 158         | Altura de la ceja derecha      | AU1, AU2, AU4                |
| `ceja_izq`              | 295 → 385        | Altura de la ceja izquierda    | AU1, AU2, AU4                |
| `ancho_boca`            | 78 → 308         | Ancho de la boca               | AU12, AU15, AU20, AU23, AU26 |
| `alto_boca`             | 13 → 14          | Apertura vertical de la boca   | AU12, AU15, AU20, AU23, AU26 |
| `entrecejo`             | 8 → 168          | Arruga entre cejas (entrecejo) | AU4                          |
| `apertura_ojo_izq`      | 159 → 145        | Apertura del ojo izquierdo     | AU5, AU7                     |
| `apertura_ojo_der`      | 386 → 374        | Apertura del ojo derecho       | AU5, AU7                     |
| `elevacion_com_izq`     | 61 → 84          | Elevación comisura izquierda   | AU12, AU15                   |
| `elevacion_com_der`     | 291 → 314        | Elevación comisura derecha     | AU12, AU15                   |
| `pliegue_nas_izq`       | 220 → 305        | Pliegue nasolabial izquierdo   | AU6                          |
| `pliegue_nas_der`       | 440 → 75         | Pliegue nasolabial derecho     | AU6                          |
| `tension_menton`        | 18 → 175         | Tensión del mentón             | AU17                         |
| `elevacion_mejilla_izq` | 116 → 117        | Elevación mejilla izquierda    | AU6                          |
| `elevacion_mejilla_der` | 345 → 346        | Elevación mejilla derecha      | AU6                          |
| `contraccion_nariz`     | 19 → 20          | Contracción de la nariz        | AU9                          |
| `tension_labio_sup`     | 0 → 17           | Tensión del labio superior     | AU23, AU24                   |
| `tension_labio_inf`     | 18 → 175         | Tensión del labio inferior     | AU23, AU24                   |


> **Ejemplo**:  
> Si `ceja_der ≤ 15` y `ceja_izq ≤ 15` -> **Cejas bajadas** (AU4)

---

### Paso 4: Clasificación de emociones

La función `detect_emotion_facs(metrics)` evalúa **combinaciones de AUs** para determinar la emoción dominante:

```python
if (elevacion_mejilla_izq >= 9 and ... and ancho_boca > 50):
    return 'Feliz', (0, 255, 255)
```

#### Lista de emociones detectadas:
| Emoción       | Color BGR       | AUs principales |
|---------------|------------------|------------------|
| Feliz         | `(0, 255, 255)`  | AU6 + AU12       |
| Sonriente     | `(255, 205, 0)`  | AU6 + AU12 (leve)|
| Tristeza      | `(0, 100, 255)`  | AU1 + AU4 + AU15 |
| Enojo         | `(0, 0, 255)`    | AU4 + AU5 + AU7 + AU23 |
| Miedo         | `(128, 0, 128)`  | AU1+2 + AU5 + AU20 |
| Asombro       | `(0, 255, 255)`  | AU1+2 + AU5 + AU26 |
| Desagrado     | `(0, 255, 0)`    | AU9 + AU15 + AU16 |
| **Neutral**   | `(255, 255, 255)`| Ninguna combinación |

---

## Visualización en Tiempo Real

### Rostro recortado (esquina inferior derecha)
Se muestra en **escala de grises** con la respectiva **malla facial** dibujada (FACEMESH_TESSELATION). Siempre en un lienzo de **200x200**.


### Etiqueta de emoción
```python
cv2.putText(frame, f"[+]: {emotion}", (int((faces[10].x * w)/1.5), int(faces[10].y * h)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
```
> Esta etiqueta se coloca sobre la frente del usuario detectado.

### Gráficas
Son un total de 7 barras horizontales (una por emoción) las cuales poseen un nivel de activación basado en **promedio normalizado de métricas relevantes** y lo cual ayuda a saber qué otras emociones está teniendo una persona y con qué intensidad.

> Cada gráfica tiene colores específicos para cada emoción.

## Estructura del Código
```text
watchme.py
├── Importación de librerías
├── Configuración de cámara (1280x720)
├── Inicialización de MediaPipe Face Mesh
├── Función: detect_emotion_facs(metrics)
├── Función: emotion_graphs(metrics, frame)
├── Bucle principal:
│   ├── Captura de frame
│   ├── Procesamiento con MediaPipe
│   ├── Recorte y escalado del rostro
│   ├── Normalización con distancia interocular
│   ├── Cálculo de 17 métricas FACS
│   ├── Detección de emoción
│   ├── Dibujo de gráficos y malla
│   └── Visualización con OpenCV
└── Liberación de recursos
```

### Dependencias:
Librerías:
- `mediapipe>=0.10.31`
- `numpy>=2.2.6`
- `opencv-python>=4.12.0.88`
- Python 3.11+
```bash
pip install -r requirements.txt
```
> Funciona en **CPU** (no requiere GPU).

Descarga el modelo de MediaPipe(Face Landmarker Model):
```bash
curl -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```
> El archivo face_landmarker.task es un modelo preentrenado, no una librería de Python.
> 
> Si el archivo ya está presente en el proyecto, este paso no es necesario.

## Uso del Programa
1. Ejecutar:
   ```bash
   python watchme.py
   ```
2. La cámara se abrirá automáticamente.
3. Mira a la cámara. Se detectará tu rostro y emoción.
4. Presiona **`q`** para salir.


## Autor y Licencia
**WatchMe** es un proyecto open-source creado como prototipo académico/demostrativo.

> **Licencia**: Apache 2.0 License
> Autor: Cristian Alexander (Crisstianpd)
> Web: https://crisstianpd.vercel.app/


Si deseas seguir desarrollando este proyecto o simplemente probarlo, eres libre de hacerlo.
