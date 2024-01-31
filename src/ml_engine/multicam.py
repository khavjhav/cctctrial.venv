import cv2
import os
import sys
import time
import pprint
from ultralytics import YOLO
import supervision as sv
from utils.coco import COCO_CLASSES
from db import client
import datetime
import torch
import threading  # <-- New Import

torch.cuda.set_device(0)
datetime_fmt = "%Y-%m-%d %H:%M:%S"

# Function that wraps your existing code
def process_feed(channel_id):
    db = client["people_counter"]
    rtsp_links_collection = db["rtsp_links"]
    in_out_counter_collection = db["in_out_counter"]

    data = rtsp_links_collection.find_one({"channel_id": channel_id})
    if data is None:
        print(f"No such channel id {channel_id} in db")
        return

    pprint.pprint(data)
    LINE_START = sv.Point(data["x1"], data["y1"])
    LINE_END = sv.Point(data["x2"], data["y2"])
    RTSP_URL = data["rtsp"]
    MODEL = YOLO("./models/yolov8n.pt")

    prev_frame_time = 0
    new_frame_time = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
    line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)

    for result in MODEL.track(source=RTSP_URL, show=False, stream=True, agnostic_nms=True):
        # ... (rest of your code remains the same)
        # ...
        line_counter.in_count, line_counter.out_count = 0, 0
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
        # print(line_counter.in_count, line_counter.out_count)
        if line_counter.in_count > 0 or line_counter.out_count > 0:
            in_out_counter_collection.insert_one(
                {
                    "channel_id": channel_id,
                    "datetime": datetime.datetime.now().strftime(datetime_fmt),
                    "in_count": line_counter.in_count,
                    "out_count": line_counter.out_count,
                }
            )

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(int(fps))
        cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python worker.py <channel_id_1> <channel_id_2>")
        exit(1)

    channel_id_1 = sys.argv[1]
    channel_id_2 = sys.argv[2]

    # Create and start threads
    thread1 = threading.Thread(target=process_feed, args=(channel_id_1,))
    thread2 = threading.Thread(target=process_feed, args=(channel_id_2,))

    thread1.start()
    thread2.start()

    # Wait for both threads to finish
    thread1.join()
    thread2.join()
