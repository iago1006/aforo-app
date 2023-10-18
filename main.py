from ultralytics import YOLO
import cv2

# load yolov8 model
model = YOLO('yolov8n.pt')

# load video
video_path = './test2.mp4'
cap = cv2.VideoCapture(video_path)

ret = True

while ret:

    ret, frame = cap.read()

    if ret:
        
        results = model.track(frame, persist=True)

        frame_ = results[0].plot()

        # Redimensionar la imagen
        scale_percent = 60  # Porcentaje de reducción (puedes ajustarlo)
        width = int(frame_.shape[1] * scale_percent / 100)
        height = int(frame_.shape[0] * scale_percent / 100)
        dim = (width, height)
        frame_ = cv2.resize(frame_, dim, interpolation=cv2.INTER_AREA)

        # visualize
        cv2.imshow('frame', frame_)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
