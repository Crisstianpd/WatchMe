# Importamos librerias
import cv2
import mediapipe as mp
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION
import math


# Preparamos Camara
cap = cv2.VideoCapture(0)
cap.set(3, 1280), cap.set(4, 720)
windowName = "WatchMe - Deteccion de personas y emociones"


# Para dibujar la mallaa 
mpDraw = mp.solutions.drawing_utils
drawConf = mpDraw.DrawingSpec(color=(255, 255, 255), thickness = 1, circle_radius = 1)


# Para detectar rostros
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)






# Buffers para suavizado por cada metrica
buffers = {
    'ceja_der': [],
    'ceja_izq': [],
    'ancho_boca': [],
    'alto_boca': [],
    'entrecejo': [],
    'apertura_ojo_izq': [],
    'apertura_ojo_der': [],
    'elevacion_com_izq': [],
    'elevacion_com_der': [],
    'pliegue_nas_izq': [],
    'pliegue_nas_der': [],
    'tension_menton': [],
    'elevacion_mejilla_izq': [],
    'elevacion_mejilla_der': [],
    'contraccion_nariz': [],
    'tension_labio_sup': [],
    'tension_labio_inf': []
}

def smooth(value, buffer_name, buffer_size=5):
    # Suavizado con buffers individuales
    if buffer_name not in buffers:
        buffers[buffer_name] = []
    
    buffers[buffer_name].append(value)
    if len(buffers[buffer_name]) > buffer_size:
        buffers[buffer_name].pop(0)
    return sum(buffers[buffer_name]) / len(buffers[buffer_name])






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
    
    
    # FELICIDAD (AU6 + AU12): Elevación de mejillas + elevación de comisuras
    if (elevacion_mejilla_izq >= 9 and elevacion_mejilla_der >= 9 and
        elevacion_com_izq > 20 and elevacion_com_der > 20 and
        ancho_boca > 23 and alto_boca >= 12 and alto_boca <= 23):
        return 'Sonriente', (255, 205, 0)
    
    elif (elevacion_mejilla_izq >= 9 and elevacion_mejilla_der >= 9 and
        elevacion_com_izq > 20 and elevacion_com_der > 20 and
        ancho_boca > 50 and alto_boca >= 0 and alto_boca < 1):
        return 'Feliz', (0, 255, 255)
    
    # TRISTEZA (AU1 + AU4 + AU15): Elevación cejas internas + descenso cejas + descenso comisuras
    elif (ceja_der >= 14 and ceja_der < 17 and ceja_izq >= 14 and ceja_izq < 17 and
          elevacion_com_izq < 21 and elevacion_com_der < 21 and
          ancho_boca < 50 and alto_boca < 18 and
          apertura_ojo_izq < 13 and apertura_ojo_der < 13):
        return 'Tristeza', (0, 100, 255)
    
    # ENOJO (AU4 + AU5 + AU7 + AU23): Descenso cejas + elevación párpados + tensión párpados + tensión labios
    elif (ceja_der <= 15 and ceja_izq <= 15 and
          entrecejo < 9 and
          ancho_boca > 35 and ancho_boca < 50 and
          alto_boca < 5 and
          tension_labio_sup >= 19 and tension_labio_inf >= 21):
        return 'Enojo', (0, 0, 255)
    
    # MIEDO (AU1+2 + AU5 + AU20): Elevación cejas + elevación párpados + estiramiento horizontal labios
    elif (ceja_der > 13 and ceja_izq > 13 and
          apertura_ojo_izq >= 8 and apertura_ojo_der >= 8 and
          ancho_boca > 40 and ancho_boca < 55 and
          alto_boca > 1 and alto_boca < 10 and
          entrecejo > 6):
        return 'Miedo', (128, 0, 128)
    
    # SORPRESA/ASOMBRO (AU1+2 + AU5 + AU26): Elevación cejas + elevación párpados + descenso mandíbula
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
    
    # NEUTRAL - Estado por defecto
    else:
        return 'Neutral', (255, 255, 255)








# Bucle principal
while True:
    activeWindows = 0
    # Actualizacion de fotogramas
    ret, frame = cap.read()
    if not ret:
        break

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frameMpResults = faceMesh.process(frameRGB)

    # Trabajar sobre los rostros detectados
    if frameMpResults.multi_face_landmarks:
        # mpDraw.draw_landmarks(frame, faces, FACEMESH_TESSELATION, drawConf, drawConf)
        for faces in frameMpResults.multi_face_landmarks:

            h, w, _ = frame.shape

            # Calculamos los bounding box a partir del landmarks(Malla)
            x_vals = [int(point.x * w) for point in faces.landmark]
            y_vals = [int(point.y * h) for point in faces.landmark]
            x_min = max(0, min(x_vals))
            x_max = min(w, max(x_vals))
            y_min = max(0, min(y_vals))
            y_max = min(h, max(y_vals))

            # Verificamos que se pueda recortar el frame
            if x_max <= x_min or y_max <= y_min:
                print("Coordenadas invalidas para recorte")
                continue

            # Recortamos la cara
            face_crop = frame[y_min:y_max, x_min:x_max]

            # ESCALADO DINAMICO EN UN LIENZO 480x480
            target_size = 480
            face_h, face_w = face_crop.shape[:2]

            # Verificamos que el recorte no este vacio
            if face_h == 0 or face_w == 0:
                print("Recorte vacio")
                continue

            scale = min(target_size / face_w, target_size / face_h)
            new_w = int(face_w * scale)
            new_h = int(face_h * scale)

            # Redimensionamos el rostro y guardamos
            resized_face = cv2.resize(face_crop, (new_w, new_h))
            # resized_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)


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


            # Datos para normalizar puntos
            eyeL = pointsList[33]
            eyeR = pointsList[263]
            ref_dist = math.hypot(eyeR[0] - eyeL[0], eyeR[1] - eyeL[1])

            
            # CALCULO DE LAS METRICAS FACS
            
            # 1. Cejas (AU1, AU2, AU4)
            x1, y1 = pointsList[65]   # Ceja derecha exterior
            x2, y2 = pointsList[158]  # Ceja derecha interior
            ceja_der = math.hypot(abs(x2 - x1), abs(y2 - y1))
            ceja_der = smooth(ceja_der / ref_dist * 100, 'ceja_der')
            
            x3, y3 = pointsList[295]  # Ceja izquierda exterior
            x4, y4 = pointsList[385]  # Ceja izquierda interior
            ceja_izq = math.hypot(abs(x4 - x3), abs(y4 - y3))
            ceja_izq = smooth(ceja_izq / ref_dist * 100, 'ceja_izq')
            
            # 2. Boca (AU12, AU15, AU20, AU23, AU26)
            x5, y5 = pointsList[78]   # Comisura izquierda
            x6, y6 = pointsList[308]  # Comisura derecha
            ancho_boca = math.hypot(abs(x6 - x5), abs(y6 - y5))
            ancho_boca = smooth(ancho_boca / ref_dist * 100, 'ancho_boca')
            
            x7, y7 = pointsList[13]   # Labio superior centro
            x8, y8 = pointsList[14]   # Labio inferior centro
            alto_boca = math.hypot(abs(x8 - x7), abs(y8 - y7))
            alto_boca = smooth(alto_boca / ref_dist * 100, 'alto_boca')
            
            # 3. Entrecejo (AU4)
            x9, y9 = pointsList[8]    # Punto superior entrecejo
            x10, y10 = pointsList[168] # Punto inferior entrecejo
            entrecejo = math.hypot(abs(x10 - x9), abs(y10 - y9))
            entrecejo = smooth(entrecejo / ref_dist * 100, 'entrecejo')
            
            # 4. Apertura de ojos (AU5, AU7)
            x_sup_izq, y_sup_izq = pointsList[159]  # Párpado superior izquierdo
            x_inf_izq, y_inf_izq = pointsList[145]  # Párpado inferior izquierdo
            apertura_ojo_izq = math.hypot(abs(x_inf_izq - x_sup_izq), abs(y_inf_izq - y_sup_izq))
            apertura_ojo_izq = smooth(apertura_ojo_izq / ref_dist * 100, 'apertura_ojo_izq')
            
            x_sup_der, y_sup_der = pointsList[386]  # Párpado superior derecho
            x_inf_der, y_inf_der = pointsList[374]  # Párpado inferior derecho
            apertura_ojo_der = math.hypot(abs(x_inf_der - x_sup_der), abs(y_inf_der - y_sup_der))
            apertura_ojo_der = smooth(apertura_ojo_der / ref_dist * 100, 'apertura_ojo_der')
            
            # 5. Elevacion de comisuras (AU12, AU15)
            x_com_izq, y_com_izq = pointsList[61]   # Comisura izquierda
            x_ref_com_izq, y_ref_com_izq = pointsList[84]   # Referencia lateral izquierda
            elevacion_com_izq = math.hypot(abs(x_ref_com_izq - x_com_izq), abs(y_ref_com_izq - y_com_izq))
            elevacion_com_izq = smooth(elevacion_com_izq / ref_dist * 100, 'elevacion_com_izq')
            
            x_com_der, y_com_der = pointsList[291]  # Comisura derecha
            x_ref_com_der, y_ref_com_der = pointsList[314]  # Referencia lateral derecha
            elevacion_com_der = math.hypot(abs(x_ref_com_der - x_com_der), abs(y_ref_com_der - y_com_der))
            elevacion_com_der = smooth(elevacion_com_der / ref_dist * 100, 'elevacion_com_der')
            
            # 6. Pliegues nasolabiales (AU6)
            x_nas_izq1, y_nas_izq1 = pointsList[220]
            x_nas_izq2, y_nas_izq2 = pointsList[305]
            pliegue_nas_izq = math.hypot(abs(x_nas_izq2 - x_nas_izq1), abs(y_nas_izq2 - y_nas_izq1))
            pliegue_nas_izq = smooth(pliegue_nas_izq / ref_dist * 100, 'pliegue_nas_izq')
            
            x_nas_der1, y_nas_der1 = pointsList[440]
            x_nas_der2, y_nas_der2 = pointsList[75]
            pliegue_nas_der = math.hypot(abs(x_nas_der2 - x_nas_der1), abs(y_nas_der2 - y_nas_der1))
            pliegue_nas_der = smooth(pliegue_nas_der / ref_dist * 100, 'pliegue_nas_der')
            
            # 7. Tension del menton (AU17)
            x_menton1, y_menton1 = pointsList[18]   # Centro del menton
            x_menton2, y_menton2 = pointsList[175]  # Punto inferior del labio
            tension_menton = math.hypot(abs(x_menton2 - x_menton1), abs(y_menton2 - y_menton1))
            tension_menton = smooth(tension_menton / ref_dist * 100, 'tension_menton')
            
            # 8. Elevacion de mejillas (AU6)
            x_mej_izq1, y_mej_izq1 = pointsList[116]  # Mejilla izquierda superior
            x_mej_izq2, y_mej_izq2 = pointsList[117]  # Mejilla izquierda inferior
            elevacion_mejilla_izq = math.hypot(abs(x_mej_izq2 - x_mej_izq1), abs(y_mej_izq2 - y_mej_izq1))
            elevacion_mejilla_izq = smooth(elevacion_mejilla_izq / ref_dist * 100, 'elevacion_mejilla_izq')
            
            x_mej_der1, y_mej_der1 = pointsList[345]  # Mejilla derecha superior
            x_mej_der2, y_mej_der2 = pointsList[346]  # Mejilla derecha inferior
            elevacion_mejilla_der = math.hypot(abs(x_mej_der2 - x_mej_der1), abs(y_mej_der2 - y_mej_der1))
            elevacion_mejilla_der = smooth(elevacion_mejilla_der / ref_dist * 100, 'elevacion_mejilla_der')
            
            # 9. Contraccion de la nariz (AU9)
            x_nariz1, y_nariz1 = pointsList[19]   # Punta de la nariz
            x_nariz2, y_nariz2 = pointsList[20]   # Base de la nariz
            contraccion_nariz = math.hypot(abs(x_nariz2 - x_nariz1), abs(y_nariz2 - y_nariz1))
            contraccion_nariz = smooth(contraccion_nariz / ref_dist * 100, 'contraccion_nariz')
            
            # 10. Tension de labios (AU23, AU24)
            x_lab_sup1, y_lab_sup1 = pointsList[0]    # Labio superior izquierdo
            x_lab_sup2, y_lab_sup2 = pointsList[17]   # Labio superior derecho
            tension_labio_sup = math.hypot(abs(x_lab_sup2 - x_lab_sup1), abs(y_lab_sup2 - y_lab_sup1))
            tension_labio_sup = smooth(tension_labio_sup / ref_dist * 100, 'tension_labio_sup')
            
            x_lab_inf1, y_lab_inf1 = pointsList[18]   # Labio inferior centro
            x_lab_inf2, y_lab_inf2 = pointsList[175]  # Labio inferior base
            tension_labio_inf = math.hypot(abs(x_lab_inf2 - x_lab_inf1), abs(y_lab_inf2 - y_lab_inf1))
            tension_labio_inf = smooth(tension_labio_inf / ref_dist * 100, 'tension_labio_inf')
            
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
            

            activeWindows += 1
            
            # Llamada a funcion para dectar emocion
            emotion, color = detect_emotion_facs(metrics)
            putText_face = f"[{activeWindows}]: {emotion}"
            cv2.putText(frame, putText_face, (int((faces.landmark[10].x * w)/1.5), int(faces.landmark[10].y * h)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            # Aparecer  nuevas ventanas
            wr_name = "Resize Face"+str(activeWindows)
            # cv2.imshow(wr_name, resized_face)

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