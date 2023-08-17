# v1

import cv2
from ultralytics  import YOLO

# rtsp_url = 'http://61.211.241.239/nphMotionJpeg?Resolution=320x240&Quality=Standard'
rtsp_url = 'rtsp://arnab:kh4vjh4v@103.205.180.214:554/Streaming/channels/1902'
model=YOLO("yolov8n.pt")


cap = cv2.VideoCapture(rtsp_url)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame=cv2.resize(frame,(640,480))
    cv2.imshow('Video Feed', frame)
    
    model.predict(frame,save=True)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



# v2

