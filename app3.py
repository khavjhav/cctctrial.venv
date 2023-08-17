import cv2
from ultralytics  import YOLO
import time

# rtsp_url = 'http://61.211.241.239/nphMotionJpeg?Resolution=320x240&Quality=Standard'
rtsp_url = 'rtsp://arnab:kh4vjh4v@103.205.180.214:554/Streaming/channels/1901'
model=YOLO("yolov8n.pt")

cap = cv2.VideoCapture(rtsp_url)


frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

font = cv2.FONT_HERSHEY_SIMPLEX
start_time = time.time()
frame_count = 0
person_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (640, 480))
    
    # Perform object detection
    results = model.predict(frame, save=False)

    # Extract detection information
    detections = results.pred[0]
    if detections is not None:
        boxes = detections[:, :4].cpu().numpy()
        scores = detections[:, 4].cpu().numpy()
        classes = detections[:, 5].cpu().numpy().astype(int)
        
        for box, score, class_id in zip(boxes, scores, classes):
            if score > 0.5 and class_id == 0:  # Assuming class_id 0 corresponds to 'person'
                x1, y1, x2, y2 = box.astype(int)
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                person_count += 1

    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Persons: {person_count}", (frame_width - 220, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    out.write(frame)
    cv2.imshow('Video Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()