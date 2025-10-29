# Importamos librerias
import cv2
import mediapipe as mp
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION
import math



# Preparamos Camara
cap = cv2.VideoCapture(0)
cap.set(3, 1280), cap.set(4, 720)
windowName = "WatchMe - Deteccion de personas y emociones"



# Para detectar rostros
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)



# Dectectar emocion en base al sistema FACS
def detect_emotion_facs(metrics):
    
    ceja_der = metrics['ceja_der']
    ceja_izq = metrics['ceja_izq']
    ancho_boca = metrics['ancho_boca']
    alto_boca = metrics['alto_boca']
    entrecejo = metrics['entrecejo']
    apertura_ojo_izq = metrics['apertura_ojo_izq']
    apertura_ojo_der = metrics['apertura_ojo_der']
    elevacion_com_izq = metrics['elevacion_com_izq']
    elevacion_com_der = metrics['elevacion_com_der']
    pliegue_nas_izq = metrics['pliegue_nas_izq']
    pliegue_nas_der = metrics['pliegue_nas_der']
    tension_menton = metrics['tension_menton']
    elevacion_mejilla_izq = metrics['elevacion_mejilla_izq']
    elevacion_mejilla_der = metrics['elevacion_mejilla_der']
    contraccion_nariz = metrics['contraccion_nariz']
    tension_labio_sup = metrics['tension_labio_sup']
    tension_labio_inf = metrics['tension_labio_inf']
    
    # print(elevacion_mejilla_izq, elevacion_mejilla_der, elevacion_com_izq, elevacion_com_der, ancho_boca, alto_boca)
    # print(ceja_der, ceja_izq, elevacion_com_izq, elevacion_com_der, ancho_boca, alto_boca, apertura_ojo_izq, apertura_ojo_der)
    # print(ceja_der, ceja_izq, entrecejo, ancho_boca, alto_boca, tension_labio_sup, tension_labio_inf)
    # print(ceja_der, ceja_izq, apertura_ojo_izq, apertura_ojo_der, ancho_boca, alto_boca, entrecejo)
    # print(ceja_der, ceja_izq, apertura_ojo_izq, apertura_ojo_der, ancho_boca, alto_boca, entrecejo)
    # print(contraccion_nariz, elevacion_com_izq, elevacion_com_der, tension_labio_inf, pliegue_nas_izq, pliegue_nas_der, ancho_boca)
    
    # FELICIDAD (AU6 + AU12): Elevacion de mejillas + elevacion de comisuras
    if (elevacion_mejilla_izq >= 9 and elevacion_mejilla_der >= 9 and
        elevacion_com_izq > 20 and elevacion_com_der > 20 and
        ancho_boca > 23 and alto_boca >= 12 and alto_boca <= 23):
        return 'Sonriente', (255, 205, 0)
    
    elif (elevacion_mejilla_izq >= 9 and elevacion_mejilla_der >= 9 and
        elevacion_com_izq > 20 and elevacion_com_der > 20 and
        ancho_boca > 50 and alto_boca >= 0 and alto_boca < 1):
        return 'Feliz', (0, 255, 255)
    
    # TRISTEZA (AU1 + AU4 + AU15): Elevacion cejas internas + descenso cejas + descenso comisuras
    elif (ceja_der >= 14 and ceja_der < 17 and ceja_izq >= 14 and ceja_izq < 17 and
        elevacion_com_izq < 21 and elevacion_com_der < 21 and
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
        elevacion_com_izq < 27 and elevacion_com_der < 27 and
        tension_labio_inf > 20 and
        pliegue_nas_izq >= 30 and pliegue_nas_der >= 30 and
        ancho_boca < 51):
        return 'Desagrado', (0, 255, 0)
    
    # NEUTRAL
    else:
        return 'Neutral', (255, 255, 255)




# Crear graficas de emociones
def emotion_graphs(metrics, frame):
    emotions = [
        ("Sonriente", (metrics['elevacion_mejilla_izq'] + metrics['elevacion_mejilla_der'] + metrics['elevacion_com_izq'] + metrics['elevacion_com_der'] + metrics['ancho_boca']+ metrics['alto_boca'])/6, 18, 29, (255,205,0)),
        ("Feliz", (metrics['elevacion_mejilla_izq'] + metrics['elevacion_mejilla_der'] + metrics['elevacion_com_izq'] + metrics['elevacion_com_der'] + metrics['ancho_boca']+ metrics['alto_boca'])/6, 20, 25, (0,255,255)),
        ("Tristeza", (metrics['ceja_der'] + metrics['ceja_izq'] + metrics['elevacion_com_izq'] + metrics['elevacion_com_der'] + metrics['ancho_boca'] + metrics['alto_boca'] + metrics['apertura_ojo_der'] + metrics['apertura_ojo_izq'])/8, 17, 16, (0,100,255)),
        ("Enojo", (metrics['tension_labio_sup'] + metrics['tension_labio_inf'] + metrics['ceja_der'] + metrics['ceja_izq'] + metrics['entrecejo'] + metrics['ancho_boca'] + metrics['alto_boca'])/7, 18, 17.5, (0,0,255)),
        ("Miedo", (metrics['apertura_ojo_izq'] + metrics['apertura_ojo_der'] + metrics['ceja_der'] + metrics['ceja_izq'] + metrics['ancho_boca'] + metrics['alto_boca'] + metrics['entrecejo'])/7, 15.5, 14.5, (128,0,128)),
        ("Asombro", (metrics['alto_boca'] + metrics['ancho_boca'] + metrics['ceja_der'] + metrics['ceja_izq'] + metrics['entrecejo'] + metrics['apertura_ojo_izq'] + metrics['apertura_ojo_der'])/7, 15, 21, (0,255,255)),
        ("Desagrado", (metrics['contraccion_nariz'] + metrics['elevacion_com_izq'] + metrics['elevacion_com_der'] + metrics['tension_labio_inf'] + metrics['pliegue_nas_izq'] + metrics['pliegue_nas_der'] + metrics['ancho_boca'])/7, 26.5, 27.5, (0,255,0)),
    ]
    
    x_base = 30
    y_base = 50
    bar_height = 20
    bar_max_width = 200
    gap = 12
    for idx, (nombre, valor, min_val, max_val, color) in enumerate(emotions):
        nivel = max(0, min(1, (valor - min_val) / (max_val - min_val)))
        bar_width = int(nivel * bar_max_width)
        x = x_base
        y = y_base + idx * (bar_height + gap)
        # Dibujar barras
        cv2.rectangle(frame, (x, y), (x + bar_max_width, y + bar_height), (20,20,20), -1)
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), color, -1)
        # Etiqueta
        cv2.putText(frame, nombre, (x + bar_max_width + 10, y + bar_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)







# Bucle principal
while True:
    # Actualizacion de fotogramas
    ret, frame = cap.read()
    if not ret:
        break
    
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frameMpResults = faceMesh.process(frameRGB)
    
    # Trabajar sobre los rostros detectados
    if frameMpResults.multi_face_landmarks:
        for faces in frameMpResults.multi_face_landmarks:
            
            h, w, _ = frame.shape
            
            # Calculamos los bounding box a partir del landmarks(Malla)
            x_vals = [int(point.x * w) for point in faces.landmark]
            y_vals = [int(point.y * h) for point in faces.landmark]
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
            
            # Redimensionamos el rostro y guardamos
            resized_face = cv2.resize(face_crop, (new_w, new_h))
            resized_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)
            
            pointsList = []
            for lm in faces.landmark:
                # Ajustamos coordenadas del landmark original al canvas
                x_rel = int((lm.x * w - x_min) * scale)
                y_rel = int((lm.y * h - y_min) * scale)
                pointsList.append((x_rel, y_rel))
            
            # Dibujar Malla
            for connection in FACEMESH_TESSELATION:
                start_idx, end_idx = connection
                if start_idx < len(faces.landmark) and end_idx < len(faces.landmark):
                    pt1 = pointsList[start_idx]
                    pt2 = pointsList[end_idx]
                    cv2.line(resized_face, pt1, pt2, (255, 255, 255), 1)
            
            
            
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
            
            
            # Datos para normalizar puntos
            eyeL = pointsList[33]
            eyeR = pointsList[263]
            ref_dist = math.hypot(eyeR[0] - eyeL[0], eyeR[1] - eyeL[1])
            
            
            
            # CALCULO DE LAS METRICAS FACS
            
            # 1. Cejas (AU1, AU2, AU4)
            x1, y1 = pointsList[65]   # Ceja derecha exterior
            x2, y2 = pointsList[158]  # Ceja derecha interior
            ceja_der = math.hypot(abs(x2 - x1), abs(y2 - y1))
            ceja_der = ceja_der / ref_dist * 100
            
            x3, y3 = pointsList[295]  # Ceja izquierda exterior
            x4, y4 = pointsList[385]  # Ceja izquierda interior
            ceja_izq = math.hypot(abs(x4 - x3), abs(y4 - y3))
            ceja_izq = ceja_izq / ref_dist * 100
            
            
            # 2. Boca (AU12, AU15, AU20, AU23, AU26)
            x5, y5 = pointsList[78]   # Comisura izquierda
            x6, y6 = pointsList[308]  # Comisura derecha
            ancho_boca = math.hypot(abs(x6 - x5), abs(y6 - y5))
            ancho_boca = ancho_boca / ref_dist * 100
            
            x7, y7 = pointsList[13]   # Labio superior centro
            x8, y8 = pointsList[14]   # Labio inferior centro
            alto_boca = math.hypot(abs(x8 - x7), abs(y8 - y7))
            alto_boca = alto_boca / ref_dist * 100
            
            # 3. Entrecejo (AU4)
            x9, y9 = pointsList[8]    # Punto superior entrecejo
            x10, y10 = pointsList[168] # Punto inferior entrecejo
            entrecejo = math.hypot(abs(x10 - x9), abs(y10 - y9))
            entrecejo = entrecejo / ref_dist * 100
            
            # 4. Apertura de ojos (AU5, AU7)
            x_sup_izq, y_sup_izq = pointsList[159]  # Párpado superior izquierdo
            x_inf_izq, y_inf_izq = pointsList[145]  # Párpado inferior izquierdo
            apertura_ojo_izq = math.hypot(abs(x_inf_izq - x_sup_izq), abs(y_inf_izq - y_sup_izq))
            apertura_ojo_izq = apertura_ojo_izq / ref_dist * 100
            
            x_sup_der, y_sup_der = pointsList[386]  # Párpado superior derecho
            x_inf_der, y_inf_der = pointsList[374]  # Párpado inferior derecho
            apertura_ojo_der = math.hypot(abs(x_inf_der - x_sup_der), abs(y_inf_der - y_sup_der))
            apertura_ojo_der = apertura_ojo_der / ref_dist * 100
            
            # 5. Elevacion de comisuras (AU12, AU15)
            x_com_izq, y_com_izq = pointsList[61]   # Comisura izquierda
            x_ref_com_izq, y_ref_com_izq = pointsList[84]   # Referencia lateral izquierda
            elevacion_com_izq = math.hypot(abs(x_ref_com_izq - x_com_izq), abs(y_ref_com_izq - y_com_izq))
            elevacion_com_izq = elevacion_com_izq / ref_dist * 100
            
            x_com_der, y_com_der = pointsList[291]  # Comisura derecha
            x_ref_com_der, y_ref_com_der = pointsList[314]  # Referencia lateral derecha
            elevacion_com_der = math.hypot(abs(x_ref_com_der - x_com_der), abs(y_ref_com_der - y_com_der))
            elevacion_com_der = elevacion_com_der / ref_dist * 100
            
            # 6. Pliegues nasolabiales (AU6)
            x_nas_izq1, y_nas_izq1 = pointsList[220]
            x_nas_izq2, y_nas_izq2 = pointsList[305]
            pliegue_nas_izq = math.hypot(abs(x_nas_izq2 - x_nas_izq1), abs(y_nas_izq2 - y_nas_izq1))
            pliegue_nas_izq = pliegue_nas_izq / ref_dist * 100
            
            x_nas_der1, y_nas_der1 = pointsList[440]
            x_nas_der2, y_nas_der2 = pointsList[75]
            pliegue_nas_der = math.hypot(abs(x_nas_der2 - x_nas_der1), abs(y_nas_der2 - y_nas_der1))
            pliegue_nas_der = pliegue_nas_der / ref_dist * 100
            
            # 7. Tension del menton (AU17)
            x_menton1, y_menton1 = pointsList[18]   # Centro del menton
            x_menton2, y_menton2 = pointsList[175]  # Punto inferior del labio
            tension_menton = math.hypot(abs(x_menton2 - x_menton1), abs(y_menton2 - y_menton1))
            tension_menton = tension_menton / ref_dist * 100
            
            # 8. Elevacion de mejillas (AU6)
            x_mej_izq1, y_mej_izq1 = pointsList[116]  # Mejilla izquierda superior
            x_mej_izq2, y_mej_izq2 = pointsList[117]  # Mejilla izquierda inferior
            elevacion_mejilla_izq = math.hypot(abs(x_mej_izq2 - x_mej_izq1), abs(y_mej_izq2 - y_mej_izq1))
            elevacion_mejilla_izq = elevacion_mejilla_izq / ref_dist * 100
            
            x_mej_der1, y_mej_der1 = pointsList[345]  # Mejilla derecha superior
            x_mej_der2, y_mej_der2 = pointsList[346]  # Mejilla derecha inferior
            elevacion_mejilla_der = math.hypot(abs(x_mej_der2 - x_mej_der1), abs(y_mej_der2 - y_mej_der1))
            elevacion_mejilla_der = elevacion_mejilla_der / ref_dist * 100
            
            # 9. Contraccion de la nariz (AU9)
            x_nariz1, y_nariz1 = pointsList[19]   # Punta de la nariz
            x_nariz2, y_nariz2 = pointsList[20]   # Base de la nariz
            contraccion_nariz = math.hypot(abs(x_nariz2 - x_nariz1), abs(y_nariz2 - y_nariz1))
            contraccion_nariz = contraccion_nariz / ref_dist * 100
            
            # 10. Tension de labios (AU23, AU24)
            x_lab_sup1, y_lab_sup1 = pointsList[0]    # Labio superior izquierdo
            x_lab_sup2, y_lab_sup2 = pointsList[17]   # Labio superior derecho
            tension_labio_sup = math.hypot(abs(x_lab_sup2 - x_lab_sup1), abs(y_lab_sup2 - y_lab_sup1))
            tension_labio_sup = tension_labio_sup / ref_dist * 100
            
            x_lab_inf1, y_lab_inf1 = pointsList[18]   # Labio inferior centro
            x_lab_inf2, y_lab_inf2 = pointsList[175]  # Labio inferior base
            tension_labio_inf = math.hypot(abs(x_lab_inf2 - x_lab_inf1), abs(y_lab_inf2 - y_lab_inf1))
            tension_labio_inf = tension_labio_inf / ref_dist * 100
            
            # Crear diccionario de metricas
            metrics = {
                'ceja_der': ceja_der,
                'ceja_izq': ceja_izq,
                'ancho_boca': ancho_boca,
                'alto_boca': alto_boca,
                'entrecejo': entrecejo,
                'apertura_ojo_izq': apertura_ojo_izq,
                'apertura_ojo_der': apertura_ojo_der,
                'elevacion_com_izq': elevacion_com_izq,
                'elevacion_com_der': elevacion_com_der,
                'pliegue_nas_izq': pliegue_nas_izq,
                'pliegue_nas_der': pliegue_nas_der,
                'tension_menton': tension_menton,
                'elevacion_mejilla_izq': elevacion_mejilla_izq,
                'elevacion_mejilla_der': elevacion_mejilla_der,
                'contraccion_nariz': contraccion_nariz,
                'tension_labio_sup': tension_labio_sup,
                'tension_labio_inf': tension_labio_inf
            }
            
            
            # Llamada a funcion para detectar emocion y crear graficas
            emotion, color = detect_emotion_facs(metrics)
            emotion_graphs(metrics, frame)
            
            putText_face = f"[+]: {emotion}"
            cv2.putText(frame, putText_face, (int((faces.landmark[10].x * w)/1.5), int(faces.landmark[10].y * h)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    try:
        cv2.imshow(windowName, frame)
    except cv2.error:
        break
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    if cv2.getWindowProperty(windowName, cv2.WND_PROP_AUTOSIZE) < 0:
        break




cap.release()
cap.destroyAllWindows()


