import cv2
import os
import time
from ultralytics import YOLO
import supervision as sv
from utils.coco import COCO_CLASSES
from dotenv import load_dotenv

load_dotenv()

CHANNEL_ID = "1902"
RTSP_URL = f"{os.getenv('RTSP_ROOT_URL')}{CHANNEL_ID}"

model = YOLO("./models/yolov8_n.onnx")

prev_frame_time = 0
new_frame_time = 0
font = cv2.FONT_HERSHEY_SIMPLEX

LINE_START = sv.Point(1280 // 2, 0)
LINE_END = sv.Point(1280 // 2, 720)

line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)


for result in model.track(source=RTSP_URL, show=False, stream=True, agnostic_nms=True):
    frame = result.orig_img
    detections = sv.Detections.from_yolov8(result)

    if result.boxes.id is not None:
        detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

    detections = detections[(detections.class_id == 0)]
    labels = [
        f"{tracker_id} {COCO_CLASSES[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, tracker_id in detections
    ]

    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
    line_counter.trigger(detections=detections)
    line_annotator.annotate(frame=frame, line_counter=line_counter)
    print(line_counter.in_count, line_counter.out_count)

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = str(int(fps))
    cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

    # cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
