import cv2
import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from utils.coco import COCO_CLASSES


rtsp = "rtsp://arnab:kh4vjh4v@103.205.180.214:554/Streaming/channels/1902"
video = 0  # "test_video.mp4"
cap = cv2.VideoCapture(rtsp)  # For Webcam

model = YOLO("./models/yolov8n.onnx")
# model.export(format="onnx")

prev_frame_time = 0
new_frame_time = 0
font = cv2.FONT_HERSHEY_SIMPLEX

while cap.isOpened():
    success, frame = cap.read()
    if success:
        frame = cv2.resize(frame, (640, 480))
        results = model.predict(source=frame, show=False, stream=True)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_name = COCO_CLASSES[int(box.cls[0])]
                if class_name != "person":
                    continue

                bbox = box.xyxy[0]
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cv2.rectangle(frame, (x, y), (w, h), (100, 255, 0), 2)
                cv2.putText(frame, class_name, (x, y), font, 1, (100, 255, 0), 2)

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(int(fps))
        cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

"""

WARNING  'Boxes.boxes' is deprecated. Use 'Boxes.data' instead.
ultralytics.engine.results.Boxes object with attributes:

boxes: tensor([[ 92.2567, 317.7429, 136.3641, 471.9507,   0.8223,   0.0000]])
cls: tensor([0.])
conf: tensor([0.8223])
data: tensor([[ 92.2567, 317.7429, 136.3641, 471.9507,   0.8223,   0.0000]])
id: None
is_track: False
orig_shape: (480, 640)
shape: torch.Size([1, 6])
xywh: tensor([[114.3104, 394.8468,  44.1074, 154.2078]])
xywhn: tensor([[0.1786, 0.8226, 0.0689, 0.3213]])
xyxy: tensor([[ 92.2567, 317.7429, 136.3641, 471.9507]])
xyxyn: tensor([[0.1442, 0.6620, 0.2131, 0.9832]])
"""
