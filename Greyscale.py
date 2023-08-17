import cv2
import numpy as np
from ultralytics import YOLO

# RTSP URL for the video stream
rtsp_url = 'rtsp://arnab:kh4vjh4v@103.205.180.214:554/Streaming/channels/1902'

# Initialize the YOLO model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(rtsp_url)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Create a pseudo RGB frame from grayscale
    pseudo_rgb_frame = np.repeat(gray_frame[:, :, np.newaxis], 3, axis=2)

    # Resize the pseudo RGB frame
    pseudo_rgb_frame = cv2.resize(pseudo_rgb_frame, (640, 480))

    cv2.imshow('Pseudo RGB Video Feed', pseudo_rgb_frame)

    # Perform object detection on the pseudo RGB frame
    model.predict(pseudo_rgb_frame, save=True)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
