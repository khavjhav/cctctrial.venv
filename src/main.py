import cv2
import time
from ultralytics import YOLO


rtsp ="rtsp://arnab:kh4vjh4v@103.205.180.214:554/Streaming/channels/1902"
cap = cv2.VideoCapture(rtsp)  # For Webcam

model = YOLO("./models/yolov8n.onnx")
# model.export(format="onnx")

prev_frame_time = 0
new_frame_time = 0
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    success, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    results = model.predict(source=frame, show=True)
    new_frame_time = time.time()
    
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = str(int(fps))
    cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break