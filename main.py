import os
import ssl
import shutil
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import jwt

app = FastAPI()

SECRET_KEY = "tu_clave_secreta_super_segura"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Base de datos simulada (la contraseña real debe estar encriptada)
db_usuarios = {
    "admin@ejemplo.com": {
        "email": "admin@ejemplo.com",
        "hashed_password": pwd_context.hash("123456") 
    }
}

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = db_usuarios.get(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Correo o contraseña incorrectos",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": user["email"]})
    return {"access_token": access_token, "token_type": "bearer"}


ssl._create_default_https_context = ssl._create_unverified_context

from base_de_datos import SessionLocal, Pitcher, AnalisisBiomecanico, Base, engine

Base.metadata.create_all(engine)

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="API de Biomecánica de Pitching Pro",
    description="Analizador biomecánico profesional para pitchers de béisbol, utilizando visión por computadora y aprendizaje automático.",
    version="3.6.0"
)

from fastapi.staticfiles import StaticFiles

# Servir archivos estáticos del frontend directamente desde la raíz
app.mount("/frontend", StaticFiles(directory="Frontend", html=True), name="frontend")
# Servir la carpeta de videos procesados
app.mount("/videos_procesados", StaticFiles(directory="videos_procesados"), name="videos_procesados")

# Permitir solicitudes desde cualquier frontend (HTML / JS / Web)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carpeta para almacenar videos procesados
OUTPUT_DIR = "videos_procesados"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# CÁLCULOS MATEMÁTICOS Y BIOMECÁNICOS
# ==========================================

def calcular_angulo_3puntos(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radianes = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angulo = np.abs(radianes * 180.0 / np.pi)
    if angulo > 180.0:
        angulo = 360.0 - angulo
    return round(float(angulo), 2)


def calcular_inclinacion_torso(hombro_izq, hombro_der, cadera_izq, cadera_der):
    mid_hombros = [(hombro_izq[0] + hombro_der[0]) / 2, (hombro_izq[1] + hombro_der[1]) / 2]
    mid_caderas = [(cadera_izq[0] + cadera_der[0]) / 2, (cadera_izq[1] + cadera_der[1]) / 2]
    dx = mid_hombros[0] - mid_caderas[0]
    dy = mid_hombros[1] - mid_caderas[1]
    return round(float(np.degrees(np.arctan2(abs(dx), abs(dy)))), 2)


def calcular_separacion_cadera_hombro(hombro_izq, hombro_der, cadera_izq, cadera_der):
    vec_hombros = np.array([hombro_der[0] - hombro_izq[0], hombro_der[1] - hombro_izq[1]])
    vec_caderas = np.array([cadera_der[0] - cadera_izq[0], cadera_der[1] - cadera_izq[1]])
    norm_h = np.linalg.norm(vec_hombros)
    norm_c = np.linalg.norm(vec_caderas)
    if norm_h == 0 or norm_c == 0:
        return 0.0
    dot_product = np.clip(np.dot(vec_hombros, vec_caderas) / (norm_h * norm_c), -1.0, 1.0)
    return round(float(np.degrees(np.arccos(dot_product))), 2)


def evaluar_metrica(valor, min_optimo, max_optimo, unidad="°", mensaje_bajo="", mensaje_alto="", mensaje_ok=""):
    if valor < min_optimo:
        estado = "Bajo / Riesgo"
        feedback = mensaje_bajo
    elif valor > max_optimo:
        estado = "Alto / Riesgo"
        feedback = mensaje_alto
    else:
        estado = "Excelente"
        feedback = mensaje_ok

    return {
        "valor": valor,
        "unidad": unidad,
        "rango_optimo": f"{min_optimo}{unidad} - {max_optimo}{unidad}",
        "estado": estado,
        "evaluacion": feedback
    }


# ==========================================
# PROCESAMIENTO Y DIBUJO SOBRE VIDEO
# ==========================================

def procesar_y_renderizar_video(input_path: str, output_path: str):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Encoder para video de salida (.mp4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frames_data = []
    temp_frames_cache = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            imagen_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultados = pose.process(imagen_rgb)

            if resultados.pose_landmarks:
                lm = resultados.pose_landmarks.landmark

                # Puntos articulares en píxeles
                h_izq = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height]
                h_der = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * width, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * height]
                c_izq = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x * width, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y * height]
                c_der = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x * width, lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y * height]
                t_izq = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * width, lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * height]
                t_der = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * width, lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * height]

                # Identificación del brazo lanzador (mano más elevada)
                if lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y < lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y:
                    hombro_lanzador, cadera_lanzador = h_der, c_der
                    codo_lanzador = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * width, lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * height]
                    muneca_lanzadora = [lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * width, lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * height]
                    tobillo_delantero = t_izq
                else:
                    hombro_lanzador, cadera_lanzador = h_izq, c_izq
                    codo_lanzador = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * width, lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * height]
                    muneca_lanzadora = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x * width, lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y * height]
                    tobillo_delantero = t_der

                ang_codo = calcular_angulo_3puntos(hombro_lanzador, codo_lanzador, muneca_lanzadora)
                ang_hombro = calcular_angulo_3puntos(cadera_lanzador, hombro_lanzador, codo_lanzador)
                inc_torso = calcular_inclinacion_torso(h_izq, h_der, c_izq, c_der)
                sep_cadera_hombro = calcular_separacion_cadera_hombro(h_izq, h_der, c_izq, c_der)
                dist_zancada = float(np.linalg.norm(np.array(t_izq) - np.array(t_der)))

                frames_data.append({
                    "angulo_codo": ang_codo,
                    "angulo_hombro": ang_hombro,
                    "inclinacion_torso": inc_torso,
                    "separacion_cadera_hombro": sep_cadera_hombro,
                    "distancia_zancada": dist_zancada,
                    "tobillo_y": tobillo_delantero[1],
                    "muneca_x": muneca_lanzadora[0]
                })

                # Dibujar esqueleto de MediaPipe
                mp_drawing.draw_landmarks(
                    frame,
                    resultados.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

                # Overlay de información en pantalla
                cv2.rectangle(frame, (10, 10), (380, 110), (0, 0, 0), -1)
                cv2.putText(frame, f"Codo: {ang_codo:.1f} deg", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Hombro: {ang_hombro:.1f} deg", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Sep. Cadera-Hombro: {sep_cadera_hombro:.1f} deg", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 155, 0), 2)

            temp_frames_cache.append(frame)

    # Identificar fases clave (Detección Cinemática Precisa)
    idx_fs = 0
    idx_rel = 0
    if frames_data:
        # Release Point: Punto de máximo avance horizontal de la muñeca
        munecas_x = [f["muneca_x"] for f in frames_data]
        idx_rel = int(np.argmax(munecas_x))

        # Foot Strike: Buscar la máxima profundidad Y del tobillo delantero ANTES del Release Point
        frames_busqueda = frames_data[:idx_rel] if idx_rel > 5 else frames_data
        
        max_y = -1
        for idx, f in enumerate(frames_busqueda):
            if f["tobillo_y"] > max_y:
                max_y = f["tobillo_y"]
                idx_fs = idx

        # Ajuste por inercia: si ya comenzó a subir tras tocar tierra
        if idx_fs > 0 and idx_fs < len(frames_data) - 1:
            if frames_data[idx_fs]["tobillo_y"] < frames_data[idx_fs - 1]["tobillo_y"]:
                idx_fs -= 1

        # Renderizar etiquetas clave en los fotogramas correspondientes
        for idx, frame in enumerate(temp_frames_cache):
            if idx == idx_fs:
                cv2.putText(frame, ">>> FOOT STRIKE <<<", (width // 2 - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            elif idx == idx_rel:
                cv2.putText(frame, ">>> RELEASE POINT <<<", (width // 2 - 170, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
            
            out.write(frame)

    cap.release()
    out.release()

    if not frames_data:
        return None, None

    return frames_data[idx_fs], frames_data[idx_rel]


# ==========================================
# ENDPOINTS
# ==========================================

@app.post("/analizar-video/{pitcher_id}")
async def analizar_video(pitcher_id: int, file: UploadFile = File(...)):
    session = SessionLocal()
    pitcher = session.query(Pitcher).filter(Pitcher.id == pitcher_id).first()

    if not pitcher:
        session.close()
        raise HTTPException(status_code=404, detail="Pitcher no encontrado en la base de datos")

    nombre_pitcher = pitcher.nombre
    id_pitcher = pitcher.id

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_input_path = f"temp_{timestamp}_{file.filename}"
    output_filename = f"pitcher_{id_pitcher}_{timestamp}.mp4"
    output_video_path = os.path.join(OUTPUT_DIR, output_filename)

    with open(temp_input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        data_fs, data_rel = await run_in_threadpool(
            procesar_y_renderizar_video, temp_input_path, output_video_path
        )
    except Exception as e:
        session.close()
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        raise HTTPException(status_code=500, detail=f"Error al procesar el video: {str(e)}")

    if os.path.exists(temp_input_path):
        os.remove(temp_input_path)

    if not data_fs or not data_rel:
        session.close()
        raise HTTPException(status_code=400, detail="No se pudieron detectar marcas corporales en el video")

    # Evaluaciones clínicas
    evaluaciones_fs = {
        "angulo_codo": evaluar_metrica(
            data_fs["angulo_codo"], 80.0, 105.0,
            mensaje_bajo="Codo demasiado cerrado; sobrecarga el UCL.",
            mensaje_alto="Codo muy retrasado (arm drag); tensión en el hombro.",
            mensaje_ok="Flexión de codo óptima."
        ),
        "angulo_hombro": evaluar_metrica(
            data_fs["angulo_hombro"], 85.0, 105.0,
            mensaje_bajo="Hombro caído por debajo del plano óptimo.",
            mensaje_alto="Elevación excesiva del hombro.",
            mensaje_ok="Abducción de hombro perfectamente alineada."
        ),
        "separacion_cadera_hombro": evaluar_metrica(
            data_fs["separacion_cadera_hombro"], 25.0, 60.0,
            mensaje_bajo="Poca separación de caderas y hombros; menor torque de salida.",
            mensaje_alto="Torsión excesiva del tronco.",
            mensaje_ok="Excelente separación cadera-hombro."
        ),
        "inclinacion_torso": evaluar_metrica(
            data_fs["inclinacion_torso"], 0.0, 20.0,
            mensaje_bajo="Torso vertical alineado.",
            mensaje_alto="Inclinación lateral excesiva.",
            mensaje_ok="Excelente balance del tronco."
        )
    }

    evaluaciones_release = {
        "angulo_codo": evaluar_metrica(
            data_rel["angulo_codo"], 120.0, 150.0,
            mensaje_bajo="Extensión incompleta en el punto de soltar.",
            mensaje_alto="Cerca de la hiperextensión brusca de codo.",
            mensaje_ok="Extensión de brazo fluida y completa."
        ),
        "inclinacion_torso": evaluar_metrica(
            data_rel["inclinacion_torso"], 10.0, 35.0,
            mensaje_bajo="Poco alcance frontal al soltar.",
            mensaje_alto="Compensación excesiva de inclinación hacia el plato.",
            mensaje_ok="Excelente extensión y transferencia de masa."
        )
    }

    resumen_diagnostico = f"FS Codo: {evaluaciones_fs['angulo_codo']['estado']} | Sep: {evaluaciones_fs['separacion_cadera_hombro']['estado']}"

    nuevo_analisis = AnalisisBiomecanico(
        pitcher_id=id_pitcher,
        fecha=datetime.now(),
        angulo_codo=data_fs["angulo_codo"],
        angulo_hombro=data_fs["angulo_hombro"],
        sugerencia=resumen_diagnostico
    )
    session.add(nuevo_analisis)
    session.commit()
    session.refresh(nuevo_analisis)
    session.close()

    return {
        "estado": "Exitoso",
        "pitcher_id": id_pitcher,
        "pitcher_nombre": nombre_pitcher,
        "analisis_id": nuevo_analisis.id,
        "video_procesado_url": f"/descargar-video/{output_filename}",
        "fases_biomecanicas": {
            "foot_strike": evaluaciones_fs,
            "release_point": evaluaciones_release
        }
    }


@app.get("/descargar-video/{filename}")
async def descargar_video(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="El archivo de video no existe")
    return FileResponse(path=file_path, media_type="video/mp4", filename=filename)