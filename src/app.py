from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from src.utils.coco import COCO_CLASSES
from ultralytics import YOLO
from pathlib import Path
import uvicorn
import cv2
import os

app = FastAPI()


CHANNEL_ID = 0
# video = f"rtsp://arnab:kh4vjh4v@103.205.180.214:554/Streaming/channels/{CHANNEL_ID}"
# cap = cv2.VideoCapture(video)
model = YOLO("./models/yolov8n.onnx")
# model.export(format="onnx")

prev_frame_time = 0
new_frame_time = 0
font = cv2.FONT_HERSHEY_SIMPLEX


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
    video = f"rtsp://arnab:kh4vjh4v@103.205.180.214:554/Streaming/channels/{channel_id}"
    cap = cv2.VideoCapture(video)
    await websocket.accept()

    try:
        while True:
            success, frame = cap.read()
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

            if not success:
                break
            else:
                _, buffer = cv2.imencode(".jpg", frame)
                await websocket.send_text("some text")
                await websocket.send_bytes(buffer.tobytes())
    except WebSocketDisconnect:
        print("Client disconnected")


# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=8000)
