import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
from base_de_datos import SessionLocal, Pitcher, AnalisisBiomecanico

# ==========================================
# 1. CÁLCULO DE ÁNGULOS VECTORIALES
# ==========================================
def calcular_angulo(a, b, c):
    """
    Calcula el ángulo interno entre tres puntos 2D (a, b, c).
    'b' es la articulación central (vértice/codo).
    """
    a = np.array(a)  # Hombro [x, y]
    b = np.array(b)  # Codo [x, y]
    c = np.array(c)  # Muñeca [x, y]

    radianes = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angulo = np.abs(radianes * 180.0 / np.pi)

    if angulo > 180.0:
        angulo = 360.0 - angulo

    return round(angulo, 2)


# ==========================================
# 2. CONFIGURACIÓN DE MEDIAPIPE Y VIDEO
# ==========================================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

video_path = "Biomecanica de pitcher.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f" Error: No se pudo abrir el archivo de video '{video_path}'. Revisa la carpeta.")
    exit()

angulos_registrados = []

# Inicializar rastreador de postura corporal
with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir a RGB para procesamiento de MediaPipe
        imagen_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imagen_rgb.flags.writeable = False

        resultados = pose.process(imagen_rgb)

        imagen_rgb.flags.writeable = True
        frame_salida = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2BGR)

        # Procesar si hay puntos anatómicos detectados
        if resultados.pose_landmarks:
            landmarks = resultados.pose_landmarks.landmark

            # Obtener muñecas de ambos brazos
            muneca_der = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            muneca_izq = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

            # Detectar cuál brazo está realizando el lanzamiento (el que tiene la muñeca más alta / Y menor)
            if muneca_der.y < muneca_izq.y:
                # Brazo Derecho
                hombro_lm = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                codo_lm = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                muneca_lm = muneca_der
                cadera_lm = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            else:
                # Brazo Izquierdo
                hombro_lm = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                codo_lm = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                muneca_lm = muneca_izq
                cadera_lm = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]

            # Evaluamos si los puntos son visibles
            if (hombro_lm.visibility > 0.65 and 
                codo_lm.visibility > 0.65 and 
                muneca_lm.visibility > 0.65):

                h, w, _ = frame.shape
                hombro = [hombro_lm.x * w, hombro_lm.y * h]
                codo = [codo_lm.x * w, codo_lm.y * h]
                muneca = [muneca_lm.x * w, muneca_lm.y * h]
                cadera_y = cadera_lm.y * h

                # Solo medir cuando la mano activa esté arriba de la cadera
                if muneca[1] < cadera_y:
                    angulo_codo = calcular_angulo(hombro, codo, muneca)

                    if 65.0 <= angulo_codo <= 175.0:
                        angulos_registrados.append(angulo_codo)

                        # Dibujar ángulo sobre el codo activo
                        codo_pos = (int(codo[0]), int(codo[1]))
                        cv2.putText(frame_salida, f"Codo: {angulo_codo} deg", 
                                    (codo_pos[0] + 10, codo_pos[1]), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                        
            # Dibujar el esqueleto anatómico sobre el jugador
            mp_drawing.draw_landmarks(
                frame_salida, 
                resultados.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS
            )

        cv2.imshow('Analisis Biomecanico - Pitcher', frame_salida)

        # Presionar 'q' para cerrar la ventana manualmente
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


# ==========================================
# 3. EVALUACIÓN Y GUARDADO EN SQLITE
# ==========================================
if angulos_registrados:
    angulo_minimo = round(float(np.min(angulos_registrados)), 2)
    angulo_promedio = round(float(np.mean(angulos_registrados)), 2)
    angulo_maximo = round(float(np.max(angulos_registrados)), 2)

    # Diagnóstico según rangos ideales biomecánicos
    if angulo_minimo < 80.0:
        sugerencia = f"Flexión excesiva en fase de carga ({angulo_minimo}°). Eleva el codo para evitar tensión en el UCL."
    elif angulo_minimo > 110.0:
        sugerencia = f"Brazo demasiado extendido en la carga ({angulo_minimo}°). Mayor flexión aumentará el torque."
    else:
        sugerencia = f"Mecánica de codo limpia. Ángulo mínimo de flexión: {angulo_minimo}° (Rango ideal)."

    session = SessionLocal()
    pitcher = session.query(Pitcher).first()

    if pitcher:
        nuevo_analisis = AnalisisBiomecanico(
            pitcher_id=pitcher.id,
            fecha=datetime.now(),
            angulo_codo=angulo_minimo,
            angulo_hombro=angulo_maximo,
            sugerencia=sugerencia
        )
        session.add(nuevo_analisis)
        session.commit()

        print("\n================ RESULTADO DE LA EVALUACIÓN ================")
        print(f" Pitcher: {pitcher.nombre}")
        print(f" Flexión mínima del codo: {angulo_minimo}°")
        print(f" Extensión máxima del codo: {angulo_maximo}°")
        print(f" Recomendación: {sugerencia}")
        print(" Base de datos SQLite actualizada correctamente.")
        print("============================================================")
    else:
        print("\n No se encontró un pitcher registrado en SQLite. Ejecuta 'base_de_datos.py' primero.")

    session.close()
else:
    print("\n No se pudieron calcular ángulos válidos en el video. Revisa la visibilidad del sujeto.")