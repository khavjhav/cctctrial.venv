from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from src.utils.coco import COCO_CLASSES
from ultralytics import YOLO
from pathlib import Path
import supervision as sv
import time
import cv2
import os

app = FastAPI()


# CHANNEL_ID = 0
# video = f"rtsp://arnab:kh4vjh4v@103.205.180.214:554/Streaming/channels/{CHANNEL_ID}"
# cap = cv2.VideoCapture(video)
model = YOLO("./models/yolov8n.onnx")
# model.export(format="onnx")

prev_frame_time = 0
new_frame_time = 0
font = cv2.FONT_HERSHEY_SIMPLEX

LINE_START = sv.Point(640 // 2, 0)
LINE_END = sv.Point(640 // 2, 360)

line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(Path(BASE_DIR, "templates")))


@app.get("/")
async def base(request: Request):
    return templates.TemplateResponse("base.html", {"request": request})


@app.get("/index")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/get-stream/{channel_id}")
async def get_stream(websocket: WebSocket, channel_id: str):
    prev_frame_time = 0
    video = f"rtsp://arnab:kh4vjh4v@103.205.180.214:554/Streaming/channels/{channel_id}"
    await websocket.accept()

    try:
        for result in model.track(
            source=video, show=False, stream=True, agnostic_nms=True
        ):
            frame = result.orig_img
            print(frame.shape)
            detections = sv.Detections.from_yolov8(result)

            if result.boxes.id is not None:
                detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

            detections = detections[(detections.class_id == 0)]
            labels = [
                f"{tracker_id} {COCO_CLASSES[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, tracker_id in detections
            ]

            frame = box_annotator.annotate(
                scene=frame, detections=detections, labels=labels
            )
            line_counter.trigger(detections=detections)
            line_annotator.annotate(frame=frame, line_counter=line_counter)
            print(line_counter.in_count, line_counter.out_count)

            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = str(int(fps))
            cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

            # if await websocket.receive_text() == "stop":
            #     break

            if frame is None:
                break
            else:
                _, buffer = cv2.imencode(".jpg", frame)
                await websocket.send_text(
                    f"{line_counter.in_count}, {line_counter.out_count}"
                )
                await websocket.send_bytes(buffer.tobytes())

    except WebSocketDisconnect:
        print("Client disconnected")


# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=8000)
