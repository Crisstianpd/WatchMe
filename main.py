# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Cristian Alexander (Crisstianpd)


# Importamos librerias
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import time



FACEMESH_MINI = set([
    # Ojo izquierdo
    (33, 133), (133, 159), (159, 145), (145, 33),

    # Ojo derecho
    (362, 263), (263, 386), (386, 374), (374, 362),

    # Boca
    (61, 291), (291, 308), (308, 78), (78, 61),

    # Cejas
    (70, 63), (63, 105), (105, 66),
    (336, 296), (296, 334), (334, 293)
])



def init_camera():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280), cap.set(4, 720)
    return cap



def create_face_landmarker():
    base_options = python.BaseOptions(model_asset_path="./models/face_landmarker.task")
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1
    )
    return vision.FaceLandmarker.create_from_options(options)



def normalized_distance(p1: tuple[int, int], p2: tuple[int, int], ref_dist: float) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1]) / ref_dist * 100



# Dectectar emocion en base al sistema FACS
def detect_emotion_facs(metrics):
    
    ceja_der = metrics['ceja_der']
    ceja_izq = metrics['ceja_izq']
    ancho_boca = metrics['ancho_boca']
    alto_boca = metrics['alto_boca']
    entrecejo = metrics['entrecejo']
    apertura_ojo_izq = metrics['apertura_ojo_izq']
    apertura_ojo_der = metrics['apertura_ojo_der']
    elevacion_comisura_izq = metrics['elevacion_comisura_izq']
    elevacion_comisura_der = metrics['elevacion_comisura_der']
    pliegue_nasolabial_izq = metrics['pliegue_nasolabial_izq']
    pliegue_nasolabial_der = metrics['pliegue_nasolabial_der']
    elevacion_mejilla_izq = metrics['elevacion_mejilla_izq']
    elevacion_mejilla_der = metrics['elevacion_mejilla_der']
    contraccion_nariz = metrics['contraccion_nariz']
    tension_labio_sup = metrics['tension_labio_sup']
    tension_labio_inf = metrics['tension_labio_inf']
    
    # print(elevacion_mejilla_izq, elevacion_mejilla_der, elevacion_comisura_izq, elevacion_comisura_der, ancho_boca, alto_boca)
    # print(ceja_der, ceja_izq, elevacion_comisura_izq, elevacion_comisura_der, ancho_boca, alto_boca, apertura_ojo_izq, apertura_ojo_der)
    # print(ceja_der, ceja_izq, entrecejo, ancho_boca, alto_boca, tension_labio_sup, tension_labio_inf)
    # print(ceja_der, ceja_izq, apertura_ojo_izq, apertura_ojo_der, ancho_boca, alto_boca, entrecejo)
    # print(ceja_der, ceja_izq, apertura_ojo_izq, apertura_ojo_der, ancho_boca, alto_boca, entrecejo)
    # print(contraccion_nariz, elevacion_comisura_izq, elevacion_comisura_der, tension_labio_inf, pliegue_nasolabial_izq, pliegue_nasolabial_der, ancho_boca)
    
    # FELICIDAD (AU6 + AU12): Elevacion de mejillas + elevacion de comisuras
    if (elevacion_mejilla_izq >= 9 and elevacion_mejilla_der >= 9 and
        elevacion_comisura_izq > 20 and elevacion_comisura_der > 20 and
        ancho_boca > 23 and alto_boca >= 12 and alto_boca <= 23):
        return 'Sonriente', (255, 205, 0)
    
    elif (elevacion_mejilla_izq >= 9 and elevacion_mejilla_der >= 9 and
        elevacion_comisura_izq > 20 and elevacion_comisura_der > 20 and
        ancho_boca > 50 and alto_boca >= 0 and alto_boca < 1):
        return 'Feliz', (0, 255, 255)
    
    # TRISTEZA (AU1 + AU4 + AU15): Elevacion cejas internas + descenso cejas + descenso comisuras
    elif (ceja_der >= 14 and ceja_der < 17 and ceja_izq >= 14 and ceja_izq < 17 and
        elevacion_comisura_izq < 21 and elevacion_comisura_der < 21 and
        ancho_boca < 50 and alto_boca < 18 and
        apertura_ojo_izq < 13 and apertura_ojo_der < 13):
        return 'Tristeza', (0, 100, 255)
    
    # ENOJO (AU4 + AU5 + AU7 + AU23): Descenso cejas + elevacion parpados + tension parpados + tension labios
    elif (ceja_der <= 15 and ceja_izq <= 15 and
        entrecejo < 9 and
        ancho_boca > 35 and ancho_boca < 50 and
        alto_boca < 5 and
        tension_labio_sup >= 19 and tension_labio_inf >= 21):
        return 'Enojo', (0, 0, 255)
    
    # MIEDO (AU1+2 + AU5 + AU20): Elevacion cejas + elevacion parpados + estiramiento horizontal labios
    elif (ceja_der > 13 and ceja_izq > 13 and
        apertura_ojo_izq >= 8 and apertura_ojo_der >= 8 and
        ancho_boca > 40 and ancho_boca < 55 and
        alto_boca > 1 and alto_boca < 10 and
        entrecejo > 6):
        return 'Miedo', (128, 0, 128)
    
    # SORPRESA/ASOMBRO (AU1+2 + AU5 + AU26): Elevacion cejas + elevacion párpados + descenso mandibula
    elif (ceja_der >= 18 and ceja_izq >= 18 and
        apertura_ojo_izq >= 5 and apertura_ojo_der >= 5 and
        ancho_boca >= 45 and ancho_boca <= 55 and
        alto_boca >= 24 and alto_boca < 32 and
        entrecejo >= 8):
        return 'Asombro', (0, 255, 255)
    
    # DESAGRADO (AU9 + AU15 + AU16): Arruga nariz + descenso comisuras + descenso labio inferior
    elif (contraccion_nariz >= 6 and
        elevacion_comisura_izq < 27 and elevacion_comisura_der < 27 and
        tension_labio_inf > 20 and
        pliegue_nasolabial_izq >= 30 and pliegue_nasolabial_der >= 30 and
        ancho_boca < 51):
        return 'Desagrado', (0, 255, 0)
    
    # NEUTRAL
    else:
        return 'Neutral', (255, 255, 255)



# Crear graficas de emociones
def emotion_graphs(metrics, frame):
    emotions = [
        ("Sonriente", (metrics['elevacion_mejilla_izq'] + metrics['elevacion_mejilla_der'] + metrics['elevacion_comisura_izq'] + metrics['elevacion_comisura_der'] + metrics['ancho_boca']+ metrics['alto_boca'])/6, 18, 29, (255,205,0)),
        ("Feliz", (metrics['elevacion_mejilla_izq'] + metrics['elevacion_mejilla_der'] + metrics['elevacion_comisura_izq'] + metrics['elevacion_comisura_der'] + metrics['ancho_boca']+ metrics['alto_boca'])/6, 20, 25, (0,255,255)),
        ("Tristeza", (metrics['ceja_der'] + metrics['ceja_izq'] + metrics['elevacion_comisura_izq'] + metrics['elevacion_comisura_der'] + metrics['ancho_boca'] + metrics['alto_boca'] + metrics['apertura_ojo_der'] + metrics['apertura_ojo_izq'])/8, 17, 16, (0,100,255)),
        ("Enojo", (metrics['tension_labio_sup'] + metrics['tension_labio_inf'] + metrics['ceja_der'] + metrics['ceja_izq'] + metrics['entrecejo'] + metrics['ancho_boca'] + metrics['alto_boca'])/7, 18, 17.5, (0,0,255)),
        ("Miedo", (metrics['apertura_ojo_izq'] + metrics['apertura_ojo_der'] + metrics['ceja_der'] + metrics['ceja_izq'] + metrics['ancho_boca'] + metrics['alto_boca'] + metrics['entrecejo'])/7, 15.5, 14.5, (128,0,128)),
        ("Asombro", (metrics['alto_boca'] + metrics['ancho_boca'] + metrics['ceja_der'] + metrics['ceja_izq'] + metrics['entrecejo'] + metrics['apertura_ojo_izq'] + metrics['apertura_ojo_der'])/7, 15, 21, (0,255,255)),
        ("Desagrado", (metrics['contraccion_nariz'] + metrics['elevacion_comisura_izq'] + metrics['elevacion_comisura_der'] + metrics['tension_labio_inf'] + metrics['pliegue_nasolabial_izq'] + metrics['pliegue_nasolabial_der'] + metrics['ancho_boca'])/7, 26.5, 27.5, (0,255,0)),
    ]
    
    x, y = 30, 50
    for name, value, min_v, max_v, color in emotions:
        level = max(0, min(1, (value - min_v) / (max_v - min_v)))
        width = int(level * 200)
        cv2.rectangle(frame, (x, y), (x+200, y+20), (30,30,30), -1)
        cv2.rectangle(frame, (x, y), (x+width, y+20), color, -1)
        cv2.putText(frame, name, (x+210, y+16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y += 30



def process_frame(frame, face_mesh):
    
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frameRGB)
    results = face_mesh.detect(image)
    
    if not results.face_landmarks:
        return frame
    
    for faces in results.face_landmarks:
        
        h, w, _ = frame.shape
        
        # Calculamos los bounding box a partir del landmarks(Malla)
        x_vals = [int(point.x * w) for point in faces]
        y_vals = [int(point.y * h) for point in faces]
        x_min = max(0, min(x_vals))
        x_max = min(w, max(x_vals))
        y_min = max(0, min(y_vals))
        y_max = min(h, max(y_vals))
            
        # Recortamos la cara
        face_crop = frame[y_min:y_max, x_min:x_max]
            
        # ESCALADO DINAMICO EN UN LIENZO NUEVO
        target_size = 200
        face_h, face_w = face_crop.shape[:2]
        
        scale = min(target_size / face_w, target_size / face_h)
        new_w = int(face_w * scale)
        new_h = int(face_h * scale)
            
        # redimensionamos el rostro y guardamos
        resized_face = cv2.resize(face_crop, (new_w, new_h))
        resized_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)
            
        pointsList = []
        for lm in faces:
            # Ajustamos coordenadas del landmark original al canvas
            x_rel = int((lm.x * w - x_min) * scale)
            y_rel = int((lm.y * h - y_min) * scale)
            pointsList.append((x_rel, y_rel))
            
        # Dibujar malla (MediaPipe moderno)
        for start_idx, end_idx in FACEMESH_MINI:
            if start_idx < len(pointsList) and end_idx < len(pointsList):
                pt1 = pointsList[start_idx]
                pt2 = pointsList[end_idx]
                cv2.line(resized_face, pt1, pt2, 255, 1)
            
        # Mostrar el rostro redimensionado en la esquina inferior derecha en escala de grises
        y_offset = frame.shape[0] - new_h
        x_offset = frame.shape[1] - new_w
        if y_offset >= 0 and x_offset >= 0:
            # Convertimos el area destino a escala de grises y luego a BGR
            gray_area = cv2.cvtColor(frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w], cv2.COLOR_BGR2GRAY)
            # Pegamos el rostro gris sobre el area gris
            gray_area[:,:] = resized_face
            # Volvemos a convertir a BGR para que el frame siga siendo de 3 canales
            frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = cv2.cvtColor(gray_area, cv2.COLOR_GRAY2BGR)
        
        points = pointsList
            
        # Dato para normalizar puntos (distancia enterocular)
        ref_dist = math.hypot(points[263][0] - points[33][0], points[263][1] - points[33][1])
        
        metrics = {
            "ceja_der": normalized_distance(points[65], points[158], ref_dist),
            "ceja_izq": normalized_distance(points[295], points[385], ref_dist),
            "ancho_boca": normalized_distance(points[78], points[308], ref_dist),
            "alto_boca": normalized_distance(points[13], points[14], ref_dist),
            "entrecejo": normalized_distance(points[8], points[168], ref_dist),
            "apertura_ojo_izq": normalized_distance(points[159], points[145], ref_dist),
            "apertura_ojo_der": normalized_distance(points[386], points[374], ref_dist),
            "elevacion_comisura_izq": normalized_distance(points[61], points[84], ref_dist),
            "elevacion_comisura_der": normalized_distance(points[291], points[314], ref_dist),
            "pliegue_nasolabial_izq": normalized_distance(points[220], points[305], ref_dist),
            "pliegue_nasolabial_der": normalized_distance(points[440], points[75], ref_dist),
            "tension_labio_sup": normalized_distance(points[0], points[17], ref_dist),
            "tension_labio_inf": normalized_distance(points[18], points[175], ref_dist),
            "elevacion_mejilla_izq": normalized_distance(points[116], points[117], ref_dist),
            "elevacion_mejilla_der": normalized_distance(points[345], points[346], ref_dist),
            "contraccion_nariz": normalized_distance(points[19], points[20], ref_dist)
        }
    
        emotion, color = detect_emotion_facs(metrics)
        emotion_graphs(metrics, frame)
        cv2.putText(frame, f"[+]: {emotion}", (int((faces[10].x * w)/1.5), int(faces[10].y * h)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    
    return frame



# =========================
# MAIN LOOP
# =========================
def main():
    cap = init_camera()
    face_mesh = create_face_landmarker()
    WINDOW_NAME = "WatchMe - Detección de Personas y Emociones"

    prev_time = 0

    # Bucle principal
    while True:
        # Actualizacion de fotogramas
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = process_frame(frame, face_mesh)
        
        # FPS
        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time)) if prev_time else 0
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {fps}", (1280 - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        
        try:
            cv2.imshow(WINDOW_NAME, frame)
        except cv2.error:
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
