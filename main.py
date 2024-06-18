from ultralytics import YOLO
import cv2
from playsound import playsound
import os
import threading
import time

# Carga el modelo
model = YOLO("yolov8n.pt")

# Define el límite máximo de personas
limite_maximo = 20

# Ruta del archivo de sonido de alerta (relativa al directorio actual del script)
alert_sound_path = os.path.join(os.path.dirname(__file__), "alert_sound.mp3")

# Verifica si el archivo de sonido existe
if not os.path.exists(alert_sound_path):
    raise FileNotFoundError(f"No se ha encontrado el archivo de sonido de alerta: {alert_sound_path}")

# Función para contar personas
def contar_personas(results):
    count = 0
    for result in results:
        boxes = result.boxes
        for box in boxes:
            if int(box.cls[0]) == 0:  # 0 es el índice para 'persona' en COCO
                count += 1
    return count

# Función para reproducir sonido en un hilo separado
def reproducir_sonido(alert_sound_path):
    playsound(alert_sound_path)

# Inicializa la captura de video desde la cámara en tiempo real
cap = cv2.VideoCapture(0)  # 0 indica la primera cámara conectada, puedes ajustarlo según el número de tu cámara

# Bandera para controlar la reproducción del sonido y almacenamiento del último conteo
alerta_emitida = False
frame_skip = 30  # Procesar cada 30 fotogramas (aproximadamente 1 segundo si el video es de 30 fps)
frame_count = 0
num_personas = 0  # Inicializa el conteo de personas

# Tiempo de inicio de la última alerta mostrada
ultimo_tiempo_alerta = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Procesa solo cada frame_skip fotogramas
    if frame_count % frame_skip == 0:
        # Realiza la detección en el fotograma actual
        results = model(frame)

        # Actualiza el conteo de personas detectadas en el fotograma actual
        num_personas = contar_personas(results)

        # Comprueba si se ha alcanzado el límite máximo
        if num_personas >= limite_maximo:
            if not alerta_emitida:
                # Reproduce el sonido de alerta en un hilo separado
                threading.Thread(target=reproducir_sonido, args=(alert_sound_path,)).start()
                alerta_emitida = True  # Evita reproducir el sonido repetidamente
                ultimo_tiempo_alerta = time.time()  # Registra el tiempo en que se mostró la alerta
        else:
            alerta_emitida = False  # Resetea la bandera si el número de personas baja del límite

    # Muestra el conteo de personas en el fotograma
    cv2.putText(frame, f'Personas: {num_personas}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Muestra la alerta si se ha alcanzado el límite máximo y dentro de los 5 segundos desde la última vez
    if alerta_emitida and (time.time() - ultimo_tiempo_alerta) <= 5:
        cv2.putText(frame, 'ALERTA: Aforo Maximo Alcanzado', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Muestra el video con las detecciones
    cv2.imshow("Video", frame)

    # Salir del bucle si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera los recursos
cap.release()
cv2.destroyAllWindows()
