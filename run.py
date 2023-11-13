import argparse
from db import init_db
import supervision as sv
from ultralytics import YOLO
from utils.coco import COCO_CLASSES
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--channel", required=True)
args = parser.parse_args()

CHANNEL_ID = args.channel

# ------------------------------
# Try to connect to the database
try:
    client = init_db()
except Exception as e:
    print("Failed to connect to the database")
    exit(1)

# ------------------------------------------
# Try to find the channel id in the database
try:
    db = client["people_counter"]
    rtsp_links_collection = db["rtsp_links"]
    in_out_counter_collection = db["in_out_counter"]
    data = rtsp_links_collection.find_one({"channel_id": CHANNEL_ID})
except Exception as e:
    print("No such channel id in db")
    exit(1)
    
# ------------------------------------------
# Try to load the model
try:
    MODEL = YOLO("./models/yolov8_n.onnx")
except Exception as e:
    print("Failed to load model")
    exit(1)
    

LINE_START = sv.Point(data["x1"], data["y1"])
LINE_END = sv.Point(data["x2"], data["y2"])
RTSP_URL = data["rtsp"]

line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)

try:
    for result in MODEL.track(source=RTSP_URL, show=False, stream=True, agnostic_nms=True):
        line_counter.in_count, line_counter.out_count = 0, 0
        frame = result.orig_img
        detections = sv.Detections.from_ultralytics(result)
        # print(detections)

        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

        detections = detections[(detections.class_id == 0)]
        labels = [
            f"{tracker_id} {COCO_CLASSES[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, tracker_id in detections
        ]

        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
        line_counter.trigger(detections=detections)
        line_annotator.annotate(frame=frame, line_counter=line_counter)
        
        # Inserting data to  database
        # print(line_counter.in_count, line_counter.out_count)
        # if line_counter.in_count > 0 or line_counter.out_count > 0:
        #     in_out_counter_collection.insert_one(
        #         {
        #             "channel_id": CHANNEL_ID,
        #             "datetime": datetime.datetime.now().strftime(datetime_fmt),
        #             "in_count": line_counter.in_count,
        #             "out_count": line_counter.out_count,
        #         }
        #     )
except Exception as e:
    print(e)